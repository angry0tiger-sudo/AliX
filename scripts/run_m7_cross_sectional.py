"""Acquire immutable public Nobitex pre-hack panels and evaluate locked M7 trials."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import ssl
import time
from http.client import HTTPSConnection, HTTPException
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HOST = "apiv2.nobitex.ir"
PREHACK_START = 1_635_712_200
PREHACK_END = 1_750_192_200
INTERVAL_SECONDS = 3_600
EXPECTED_ROWS = 31_801
BTC_SYMBOL = "BTCUSDT"
CANDIDATE_SYMBOLS = (
    "ETHUSDT",
    "BNBUSDT",
    "ADAUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "LINKUSDT",
    "DOTUSDT",
)
M7_FIVE_SYMBOLS = CANDIDATE_SYMBOLS[:5]
M7_TEN_SYMBOLS = CANDIDATE_SYMBOLS
FEATURE_NAMES = (
    "btc_return_4h",
    "cross_sectional_median_return_4h_ex_target",
    "beta_24h_to_btc",
    "relative_volume_shock_24h",
)
COST_SCENARIOS_BPS = (50, 90)
MAX_CANDLES_PER_REQUEST = 400
MAX_RETRIES = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(path: str, params: Mapping[str, str] | None = None) -> dict[str, Any]:
    query = "?" + urlencode(params) if params else ""
    last_error = "unknown"
    for attempt in range(MAX_RETRIES):
        connection = HTTPSConnection(
            HOST, timeout=20, context=ssl.create_default_context()
        )
        try:
            connection.request(
                "GET",
                path + query,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "AliX-M7-Public-Market-Research/1.0",
                },
            )
            response = connection.getresponse()
            body = response.read()
            if response.status == 200:
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("m7_public_api_payload_not_object")
                return payload
            if response.status not in (408, 425, 429, 500, 502, 503, 504):
                raise RuntimeError(f"m7_public_api_http_{response.status}")
            last_error = f"http_{response.status}"
        except (HTTPException, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = type(exc).__name__
        finally:
            connection.close()
        if attempt < MAX_RETRIES - 1:
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"m7_public_api_failed:{last_error}")


def _normalize_market_key(value: str) -> str:
    return "".join(char for char in value.upper() if char.isalnum())


def _market_stats_snapshot() -> dict[str, dict[str, Any]]:
    payload = _request_json("/market/stats")
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise ValueError("m7_market_stats_missing")
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in stats.items():
        if isinstance(value, dict):
            normalized[_normalize_market_key(str(key))] = value
    return normalized


def _is_market_closed(market: Mapping[str, Any]) -> bool:
    for key in ("isClosed", "is_closed", "closed"):
        if market.get(key) is True:
            return True
    for key in ("isActive", "is_active", "active"):
        if market.get(key) is False:
            return True
    return False


def _candle_arrays(payload: Mapping[str, Any]) -> tuple[list[Any], ...]:
    status = payload.get("s")
    if status == "no_data":
        return ([], [], [], [], [], [])
    if status != "ok":
        raise ValueError("m7_udf_status_not_ok")
    names = ("t", "o", "h", "l", "c", "v")
    arrays = [payload.get(name) for name in names]
    if any(not isinstance(value, list) for value in arrays):
        raise ValueError("m7_udf_arrays_missing")
    sizes = {len(value) for value in arrays if isinstance(value, list)}
    if len(sizes) != 1:
        raise ValueError("m7_udf_array_length_mismatch")
    return tuple(arrays)  # type: ignore[return-value]


def _fetch_one_page(symbol: str, start: int, end: int) -> dict[int, dict[str, Any]]:
    payload = _request_json(
        "/market/udf/history",
        {"symbol": symbol, "resolution": "60", "from": str(start), "to": str(end + 1)},
    )
    arrays = _candle_arrays(payload)
    timestamps, opens, highs, lows, closes, volumes = arrays
    result: dict[int, dict[str, Any]] = {}
    for index, raw_stamp in enumerate(timestamps):
        try:
            stamp = int(raw_stamp)
            opened, high, low, close, volume = (
                float(opens[index]),
                float(highs[index]),
                float(lows[index]),
                float(closes[index]),
                float(volumes[index]),
            )
        except (TypeError, ValueError, OverflowError):
            raise ValueError("m7_udf_candle_value_invalid") from None
        if stamp < start or stamp > end:
            continue
        if (stamp - PREHACK_START) % INTERVAL_SECONDS:
            raise ValueError("m7_udf_timestamp_off_grid")
        if not all(
            math.isfinite(value) for value in (opened, high, low, close, volume)
        ):
            raise ValueError("m7_udf_non_finite_candle")
        if min(opened, high, low, close) <= 0 or volume < 0:
            raise ValueError("m7_udf_price_or_volume_out_of_range")
        if high < max(opened, close) or low > min(opened, close) or high < low:
            raise ValueError("m7_udf_ohlc_invariant_failed")
        if stamp in result:
            raise ValueError("m7_udf_duplicate_timestamp")
        result[stamp] = {
            "timestamp": stamp,
            "date": datetime.fromtimestamp(stamp, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    return result


def fetch_exact_prehack_history(symbol: str) -> list[dict[str, Any]]:
    expected = list(range(PREHACK_START, PREHACK_END + 1, INTERVAL_SECONDS))
    rows_by_stamp: dict[int, dict[str, Any]] = {}
    cursor = PREHACK_START
    while cursor <= PREHACK_END:
        stop = min(
            cursor + (MAX_CANDLES_PER_REQUEST - 1) * INTERVAL_SECONDS, PREHACK_END
        )
        page = _fetch_one_page(symbol, cursor, stop)
        for stamp, row in page.items():
            if stamp in rows_by_stamp:
                raise ValueError(f"m7_duplicate_across_pages:{symbol}:{stamp}")
            rows_by_stamp[stamp] = row
        cursor = stop + INTERVAL_SECONDS
        if cursor <= PREHACK_END:
            time.sleep(1.1)
    missing = [stamp for stamp in expected if stamp not in rows_by_stamp]
    unexpected = [stamp for stamp in rows_by_stamp if stamp not in set(expected)]
    if missing or unexpected or len(rows_by_stamp) != EXPECTED_ROWS:
        raise ValueError(
            f"m7_exact_coverage_failed:symbol={symbol}:expected={EXPECTED_ROWS}:"
            f"actual={len(rows_by_stamp)}:missing={len(missing)}:"
            f"first_missing={missing[0] if missing else None}:"
            f"last_missing={missing[-1] if missing else None}:unexpected={len(unexpected)}"
        )
    return [rows_by_stamp[stamp] for stamp in expected]


def _write_symbol_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("date", "open", "high", "low", "close", "volume")
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: row[name]
                    for name in ("date", "open", "high", "low", "close", "volume")
                }
            )
    temporary.replace(path)
    return sha256_file(path)


def acquire_dataset(output_dir: Path, commit_sha: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    market_stats = _market_stats_snapshot()
    statuses: dict[str, dict[str, Any]] = {}
    for symbol in CANDIDATE_SYMBOLS:
        market = market_stats.get(_normalize_market_key(symbol))
        if market is None:
            statuses[symbol] = {"status": "NOT_LISTED_IN_PUBLIC_MARKET_STATS"}
            continue
        if _is_market_closed(market):
            statuses[symbol] = {"status": "MARKET_CLOSED"}
            continue
        try:
            rows = fetch_exact_prehack_history(symbol)
            csv_path = output_dir / f"{symbol.lower()}_1h.csv"
            digest = _write_symbol_csv(rows, csv_path)
            manifest = {
                "schema_version": "alix.m7.public-prehack-ohlcv.v1",
                "symbol": symbol,
                "venue": "Nobitex",
                "market_type": "Spot",
                "interval": "1h",
                "source": "Nobitex public UDF history API",
                "endpoint": "/market/udf/history",
                "start_utc": rows[0]["date"],
                "end_utc": rows[-1]["date"],
                "start_timestamp": PREHACK_START,
                "end_timestamp": PREHACK_END,
                "row_count": len(rows),
                "expected_rows": EXPECTED_ROWS,
                "missing_intervals": 0,
                "csv_file": csv_path.name,
                "csv_sha256": digest,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "acquisition_commit_sha": commit_sha,
                "new_data_acquisition": True,
                "public_api_only": True,
                "credentials_used": False,
                "private_api_calls": 0,
                "real_order_attempts": 0,
                "paper_orders": 0,
                "holdout_bytes_read": False,
                "posthack_data_read": False,
                "data_scope": "PRE_HACK_EXPLORATORY_ONLY",
                "trading_authorization": "NO-GO",
            }
            manifest_path = output_dir / f"{symbol.lower()}_1h.manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            statuses[symbol] = {
                "status": "QUALIFIED",
                "row_count": len(rows),
                "csv_sha256": digest,
                "manifest_file": manifest_path.name,
                "market_stats_fields": sorted(market.keys()),
            }
        except (RuntimeError, ValueError, OSError) as exc:
            statuses[symbol] = {
                "status": "REJECTED_DATA_QUALITY_OR_API",
                "reason": str(exc),
            }
    qualified = [
        symbol
        for symbol in CANDIDATE_SYMBOLS
        if statuses[symbol]["status"] == "QUALIFIED"
    ]
    report = {
        "schema_version": "alix.m7.data-qualification.v1",
        "commit_sha": commit_sha,
        "venue": "Nobitex",
        "market_type": "Spot",
        "interval": "1h",
        "data_scope": "PRE_HACK_ONLY",
        "start_timestamp": PREHACK_START,
        "end_timestamp": PREHACK_END,
        "start_utc": datetime.fromtimestamp(PREHACK_START, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "end_utc": datetime.fromtimestamp(PREHACK_END, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "expected_rows_per_symbol": EXPECTED_ROWS,
        "public_market_stats_symbols": sorted(market_stats.keys()),
        "symbol_status": statuses,
        "qualified_symbols": qualified,
        "trial_data_status": {
            "NOBITEX-M7-5PAIR-RIDGE-TOP2-01": "QUALIFIED"
            if all(s in qualified for s in M7_FIVE_SYMBOLS)
            else "BLOCKED_DATA",
            "NOBITEX-M7-10PAIR-RIDGE-TOP2-01": "QUALIFIED"
            if all(s in qualified for s in M7_TEN_SYMBOLS)
            else "BLOCKED_DATA",
        },
        "new_data_acquisition": True,
        "private_api_calls": 0,
        "credentials_used": False,
        "paper_orders": 0,
        "real_order_attempts": 0,
        "holdout_bytes_read": False,
        "posthack_data_read": False,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }
    (output_dir / "m7_data_qualification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def select_top_two(predictions: Mapping[str, float]) -> list[str]:
    """Select at most two strictly positive finite predictions, ties by symbol."""
    eligible = [
        (symbol, float(score))
        for symbol, score in predictions.items()
        if math.isfinite(float(score)) and float(score) > 0.0
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return [symbol for symbol, _ in eligible[:2]]


def portfolio_interval_return(
    selected_symbols: Sequence[str],
    realized_returns: Mapping[str, float],
    cost_bps: float,
) -> tuple[float, list[float]]:
    """Equal-weight up to two positions; unused weight remains in USDT cash."""
    if (
        cost_bps < 0
        or len(selected_symbols) > 2
        or len(set(selected_symbols)) != len(selected_symbols)
    ):
        raise ValueError("m7_invalid_selected_symbol_set")
    if not selected_symbols:
        return 0.0, []
    cost_fraction = cost_bps / 10_000.0
    trade_pnls: list[float] = []
    for symbol in selected_symbols:
        if symbol not in realized_returns:
            raise ValueError(f"m7_missing_realized_return:{symbol}")
        raw_return = float(realized_returns[symbol])
        if not math.isfinite(raw_return) or raw_return <= -1.0:
            raise ValueError(f"m7_invalid_realized_return:{symbol}")
        trade_pnls.append(0.5 * (raw_return - cost_fraction))
    return float(sum(trade_pnls)), trade_pnls


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ("date", "open", "high", "low", "close", "volume")
    if tuple(frame.columns) != required or len(frame) != EXPECTED_ROWS:
        raise ValueError(f"m7_csv_schema_or_row_count_mismatch:{path.name}")
    frame["timestamp"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    if frame["timestamp"].duplicated().any():
        raise ValueError(f"m7_csv_duplicate_dates:{path.name}")
    if not (frame["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError(f"m7_csv_hourly_grid_mismatch:{path.name}")
    numeric = frame.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(numeric).all()
        or (numeric[:, :4] <= 0).any()
        or (numeric[:, 4] < 0).any()
    ):
        raise ValueError(f"m7_csv_numeric_quality_failed:{path.name}")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise ValueError(f"m7_csv_high_invariant:{path.name}")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"m7_csv_low_invariant:{path.name}")
    return frame


def load_locked_data(data_dir: Path, btc_csv: Path) -> dict[str, pd.DataFrame]:
    expected_btc_hash = (
        "0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e"
    )
    if sha256_file(btc_csv) != expected_btc_hash:
        raise ValueError("m7_btc_source_hash_mismatch")
    data: dict[str, pd.DataFrame] = {BTC_SYMBOL: _load_csv(btc_csv)}
    manifest = json.loads(
        (data_dir / "m7_data_qualification.json").read_text(encoding="utf-8")
    )
    if manifest.get("data_scope") != "PRE_HACK_ONLY":
        raise ValueError("m7_data_scope_mismatch")
    if (
        manifest.get("start_timestamp") != PREHACK_START
        or manifest.get("end_timestamp") != PREHACK_END
    ):
        raise ValueError("m7_data_range_mismatch")
    for symbol in manifest.get("qualified_symbols", []):
        info = manifest["symbol_status"][symbol]
        csv_path = data_dir / f"{symbol.lower()}_1h.csv"
        sidecar = json.loads(
            (data_dir / f"{symbol.lower()}_1h.manifest.json").read_text(
                encoding="utf-8"
            )
        )
        digest = sha256_file(csv_path)
        if digest != info["csv_sha256"] or digest != sidecar["csv_sha256"]:
            raise ValueError(f"m7_symbol_hash_mismatch:{symbol}")
        if sidecar.get("row_count") != EXPECTED_ROWS:
            raise ValueError(f"m7_symbol_manifest_row_count:{symbol}")
        data[symbol] = _load_csv(csv_path)
    reference_times = data[BTC_SYMBOL]["timestamp"]
    for symbol, frame in data.items():
        if not frame["timestamp"].equals(reference_times):
            raise ValueError(f"m7_cross_asset_timestamp_mismatch:{symbol}")
    return data


def _log_returns(closes: np.ndarray) -> np.ndarray:
    result = np.full(closes.shape, np.nan, dtype=float)
    result[1:] = np.log(closes[1:] / closes[:-1])
    return result


def build_panel(
    data: Mapping[str, pd.DataFrame], symbols: Sequence[str]
) -> pd.DataFrame:
    if len(symbols) < 2 or any(symbol not in data for symbol in symbols):
        raise ValueError("m7_unqualified_candidate_universe")
    btc = data[BTC_SYMBOL]
    btc_close = btc["close"].to_numpy(dtype=float)
    btc_returns = _log_returns(btc_close)
    close_by_symbol = {
        symbol: data[symbol]["close"].to_numpy(dtype=float) for symbol in symbols
    }
    open_by_symbol = {
        symbol: data[symbol]["open"].to_numpy(dtype=float) for symbol in symbols
    }
    volume_by_symbol = {
        symbol: data[symbol]["volume"].to_numpy(dtype=float) for symbol in symbols
    }
    returns_by_symbol = {
        symbol: _log_returns(close_by_symbol[symbol]) for symbol in symbols
    }
    dates = btc["date"].astype(str).to_numpy()
    records: list[dict[str, Any]] = []
    for decision_index in range(24, EXPECTED_ROWS - 5, 4):
        btc_return_4h = math.log(
            btc_close[decision_index] / btc_close[decision_index - 4]
        )
        for symbol in symbols:
            peer_returns = [
                math.log(
                    close_by_symbol[peer][decision_index]
                    / close_by_symbol[peer][decision_index - 4]
                )
                for peer in symbols
                if peer != symbol
            ]
            asset_window = returns_by_symbol[symbol][
                decision_index - 23 : decision_index + 1
            ]
            btc_window = btc_returns[decision_index - 23 : decision_index + 1]
            if len(asset_window) != 24 or len(btc_window) != 24:
                continue
            if not np.isfinite(asset_window).all() or not np.isfinite(btc_window).all():
                continue
            btc_variance = float(np.var(btc_window, ddof=1))
            if not math.isfinite(btc_variance) or btc_variance <= 1e-12:
                continue
            beta = float(np.cov(asset_window, btc_window, ddof=1)[0, 1] / btc_variance)
            prior_volume = volume_by_symbol[symbol][
                decision_index - 24 : decision_index
            ]
            median_volume = float(np.median(prior_volume))
            current_volume = float(volume_by_symbol[symbol][decision_index])
            if not math.isfinite(median_volume) or median_volume <= 0.0:
                continue
            relative_volume_shock = math.log1p(current_volume / median_volume)
            peer_median = float(np.median(peer_returns))
            target = math.log(
                open_by_symbol[symbol][decision_index + 5]
                / open_by_symbol[symbol][decision_index + 1]
            )
            features = (btc_return_4h, peer_median, beta, relative_volume_shock)
            if not all(math.isfinite(value) for value in (*features, target)):
                continue
            records.append(
                {
                    "decision_index": decision_index,
                    "decision_time": dates[decision_index],
                    "symbol": symbol,
                    "btc_return_4h": btc_return_4h,
                    "cross_sectional_median_return_4h_ex_target": peer_median,
                    "beta_24h_to_btc": beta,
                    "relative_volume_shock_24h": relative_volume_shock,
                    "target_return_4h": target,
                }
            )
    panel = pd.DataFrame.from_records(records)
    if panel.empty:
        raise ValueError("m7_panel_empty_after_feature_qualification")
    return panel


def _drawdown(returns: Sequence[float]) -> float:
    nav = 1.0
    peak = 1.0
    maximum = 0.0
    for value in returns:
        nav *= 1.0 + float(value)
        peak = max(peak, nav)
        maximum = max(maximum, (peak - nav) / peak)
    return maximum


def _metrics(
    trade_pnls: Sequence[float], portfolio_returns: Sequence[float]
) -> dict[str, Any]:
    positives = [float(value) for value in trade_pnls if value > 0.0]
    negatives = [float(value) for value in trade_pnls if value < 0.0]
    denominator = abs(sum(negatives))
    profit_factor = float(sum(positives) / denominator) if denominator > 0.0 else None
    portfolio_return = (
        float(np.prod([1.0 + value for value in portfolio_returns]) - 1.0)
        if portfolio_returns
        else 0.0
    )
    return {
        "completed_oos_trades": len(trade_pnls),
        "net_profit_factor": profit_factor,
        "net_pnl_fraction_initial_equity": float(sum(trade_pnls)),
        "portfolio_return_fraction_initial_equity": portfolio_return,
        "max_drawdown": _drawdown(portfolio_returns),
    }


def _evaluate_fold(
    model: Pipeline, test_panel: pd.DataFrame, cost_bps: int, fold_id: str
) -> tuple[dict[str, Any], list[float], list[float]]:
    trade_pnls: list[float] = []
    portfolio_returns: list[float] = []
    trades_by_symbol: dict[str, int] = {}
    for _, group in test_panel.groupby("decision_index", sort=True):
        predictions = model.predict(group.loc[:, FEATURE_NAMES])
        scores = {
            str(symbol): float(score)
            for symbol, score in zip(group["symbol"], predictions, strict=True)
        }
        selected = select_top_two(scores)
        realized = {
            str(row.symbol): float(row.target_return_4h)
            for row in group.itertuples(index=False)
        }
        interval_return, contributions = portfolio_interval_return(
            selected, realized, cost_bps
        )
        portfolio_returns.append(interval_return)
        trade_pnls.extend(contributions)
        for symbol in selected:
            trades_by_symbol[symbol] = trades_by_symbol.get(symbol, 0) + 1
    metrics = _metrics(trade_pnls, portfolio_returns)
    valid = metrics["completed_oos_trades"] >= 20
    stable = bool(
        valid
        and metrics["net_pnl_fraction_initial_equity"] > 0.0
        and metrics["net_profit_factor"] is not None
        and metrics["net_profit_factor"] > 1.0
    )
    return (
        {
            "fold_id": fold_id,
            **metrics,
            "valid": valid,
            "stable": stable,
            "trades_by_symbol": trades_by_symbol,
        },
        trade_pnls,
        portfolio_returns,
    )


def _run_candidate(
    panel: pd.DataFrame, symbols: Sequence[str], trial_id: str
) -> dict[str, Any]:
    unique_indices = sorted(int(value) for value in panel["decision_index"].unique())
    blocks = [list(map(int, block)) for block in np.array_split(unique_indices, 5)]
    models: dict[str, Pipeline] = {}
    tests: dict[str, pd.DataFrame] = {}
    fold_train_counts: dict[str, int] = {}
    fold_test_counts: dict[str, int] = {}
    for fold_number in range(1, 5):
        prior_blocks = blocks[:fold_number]
        test_indices = blocks[fold_number]
        fold_id = f"WF-0{fold_number}"
        if not test_indices:
            continue
        test_start = min(test_indices)
        next_start = (
            min(blocks[fold_number + 1])
            if fold_number < 4 and blocks[fold_number + 1]
            else EXPECTED_ROWS - 1
        )
        prior_indices = [value for block in prior_blocks for value in block]
        train = panel[
            panel["decision_index"].isin(prior_indices)
            & (panel["decision_index"] + 5 <= test_start)
        ]
        test = panel[
            panel["decision_index"].isin(test_indices)
            & (panel["decision_index"] + 5 <= next_start)
        ]
        fold_train_counts[fold_id] = int(len(train))
        fold_test_counts[fold_id] = int(len(test))
        if len(train) < 200 or test.empty:
            continue
        model = Pipeline(
            [("standard_scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
        )
        model.fit(train.loc[:, FEATURE_NAMES], train["target_return_4h"])
        models[fold_id] = model
        tests[fold_id] = test.copy()
    if not models:
        return {
            "trial_id": trial_id,
            "status": "INSUFFICIENT_SAMPLE",
            "symbol_universe": list(symbols),
            "panel_rows": int(len(panel)),
            "fold_train_rows": fold_train_counts,
            "fold_test_rows": fold_test_counts,
            "fit_count": 0,
        }
    cost_results: dict[str, Any] = {}
    overall_pass = True
    for cost_bps in COST_SCENARIOS_BPS:
        fold_results: list[dict[str, Any]] = []
        pooled_trades: list[float] = []
        pooled_returns: list[float] = []
        for fold_id in ("WF-01", "WF-02", "WF-03", "WF-04"):
            if fold_id not in models:
                fold_results.append(
                    {
                        "fold_id": fold_id,
                        "status": "INSUFFICIENT_SAMPLE",
                        "valid": False,
                        "stable": False,
                    }
                )
                continue
            result, trades, returns = _evaluate_fold(
                models[fold_id], tests[fold_id], cost_bps, fold_id
            )
            fold_results.append(result)
            pooled_trades.extend(trades)
            pooled_returns.extend(returns)
        pooled = _metrics(pooled_trades, pooled_returns)
        valid_folds = sum(bool(item.get("valid")) for item in fold_results)
        stable_folds = sum(bool(item.get("stable")) for item in fold_results)
        passed = bool(
            pooled["net_profit_factor"] is not None
            and pooled["net_profit_factor"] >= 1.10
            and pooled["completed_oos_trades"] >= 100
            and pooled["max_drawdown"] <= 0.15
            and valid_folds >= 3
            and stable_folds >= 3
        )
        overall_pass = overall_pass and passed
        cost_results[f"Cost-{cost_bps}"] = {
            **pooled,
            "valid_folds": valid_folds,
            "stable_folds": stable_folds,
            "fold_results": fold_results,
            "economic_gates_passed": passed,
        }
    return {
        "trial_id": trial_id,
        "status": "ECONOMIC_PASS" if overall_pass else "NO-GO",
        "symbol_universe": list(symbols),
        "panel_rows": int(len(panel)),
        "fold_train_rows": fold_train_counts,
        "fold_test_rows": fold_test_counts,
        "fit_count": len(models),
        "model": "StandardScaler + Ridge(alpha=1.0)",
        "feature_names": list(FEATURE_NAMES),
        "cost_scenarios": cost_results,
        "economic_gates_passed": overall_pass,
    }


def evaluate_m7(data_dir: Path, btc_csv: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config["preregistration"]["status"] != "PREREGISTERED_LOCKED"
        or config["preregistration"]["logical_trial_count"] != 2
    ):
        raise ValueError("m7_preregistration_not_locked_or_trial_count_mismatch")
    data = load_locked_data(data_dir, btc_csv)
    results: dict[str, Any] = {}
    for trial in config["trials"]:
        trial_id = trial["trial_id"]
        symbols = trial["symbols"]
        missing = [symbol for symbol in symbols if symbol not in data]
        if missing:
            results[trial_id] = {
                "trial_id": trial_id,
                "status": "BLOCKED_DATA",
                "symbol_universe": symbols,
                "missing_symbols": missing,
            }
            continue
        try:
            panel = build_panel(data, symbols)
            results[trial_id] = _run_candidate(panel, symbols, trial_id)
        except (ValueError, KeyError, FloatingPointError) as exc:
            results[trial_id] = {
                "trial_id": trial_id,
                "status": "BLOCKED_DATA_OR_FEATURE_QUALITY",
                "symbol_universe": symbols,
                "reason": str(exc),
            }
    overall = (
        "ECONOMIC_PASS"
        if all(item["status"] == "ECONOMIC_PASS" for item in results.values())
        else "NO-GO"
    )
    return {
        "schema_version": "alix.m7.cross-sectional-evaluation.v1",
        "phase": "M7",
        "preregistration_sha256": sha256_file(config_path),
        "btc_source_sha256": sha256_file(btc_csv),
        "data_scope": "PRE_HACK_EXPLORATORY_ONLY",
        "target": "4h forward log return from next-hour open to open four hours later",
        "model": "StandardScaler + Ridge(alpha=1.0)",
        "trials": results,
        "economic_acceptance": {
            "overall_verdict": overall,
            "net_profit_factor_min": 1.10,
            "completed_oos_trades_min": 100,
            "max_drawdown_fraction_max": 0.15,
            "stable_folds_min": 3,
            "valid_folds_min": 3,
        },
        "safety": {
            "new_data_acquisition_during_evaluation": False,
            "holdout_bytes_read": False,
            "posthack_data_read": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "paper_orders": 0,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/nobitex/prehack/m7")
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/nobitex/prehack/m7")
    )
    parser.add_argument(
        "--btc-csv",
        type=Path,
        default=Path("data/nobitex/prehack/nobitex-btcusdt-1h.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/m7/cross_sectional_preregistration.json"),
    )
    parser.add_argument(
        "--report-out", type=Path, default=Path("m7_evaluation_report.json")
    )
    parser.add_argument("--commit-sha", default="UNKNOWN")
    args = parser.parse_args()
    if args.acquire == args.evaluate:
        parser.error("select exactly one of --acquire or --evaluate")
    if args.acquire:
        report = acquire_dataset(args.output_dir, args.commit_sha)
    else:
        report = evaluate_m7(args.data_dir, args.btc_csv, args.config)
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
