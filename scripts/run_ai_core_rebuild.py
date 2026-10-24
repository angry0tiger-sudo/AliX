"""Run the isolated 4H AI-core research rebuild; never creates orders or approvals."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
import zipfile

import numpy as np
import pandas as pd
import urllib3
from pandas import DataFrame

from scripts import run_stage_b_triple_barrier_orderbook as stage_b
from services.ai_core.ensemble import (
    EnsembleConfig,
    fit_weighted_ensemble,
    regression_metrics,
)
from services.ai_core.features import (
    AI_FEATURE_COLUMNS,
    AI_FEATURE_SCHEMA_VERSION,
    aggregate_hourly_to_4h,
    build_ai_features,
)
from services.ai_core.regime import infer_hmm_regimes
from services.ai_core.strategies import (
    ROUND_TRIP_COST_BPS,
    generate_candidate_signal,
    route_research_regime,
)
from services.strategy_selection.hmm_training import (
    calculate_regime_features,
    fit_gaussian_hmm,
)
from services.strategy_selection.registry import load_registry_document

SYMBOL = "ETHUSDT"
INTERVAL = "1h"
ARCHIVE_ROOT = "https://data.binance.vision/data/spot"
OUTPUT_SCHEMA = "alix.ai-core-rebuild-report.v1"
FORWARD_HORIZON_4H = 6
ROUND_TRIP_COST = ROUND_TRIP_COST_BPS / 10_000.0
MINIMUM_TRAIN_ROWS = 360
FOLD_COUNT = 3
HTTP_POOL = urllib3.PoolManager()


def _fetch_archive(url: str) -> bytes | None:
    source = urlsplit(url)
    if (
        source.scheme != "https"
        or source.hostname != "data.binance.vision"
        or not source.path.startswith("/data/spot/")
    ):
        raise ValueError("ai_core_archive_url_not_allowlisted")
    try:
        response = HTTP_POOL.request(
            "GET",
            url,
            headers={"User-Agent": "AliX-ai-core-research/1.0"},
            timeout=urllib3.Timeout(connect=10.0, read=60.0),
            retries=False,
            redirect=False,
        )
    except urllib3.exceptions.HTTPError as error:
        raise RuntimeError(f"ai_core_archive_download_failed:{error}") from error
    try:
        if response.status == 404:
            return None
        if response.status != 200:
            raise RuntimeError(f"ai_core_archive_http_status:{response.status}")
        return response.data
    finally:
        response.release_conn()


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    current = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    last = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    months: list[pd.Timestamp] = []
    while current <= last:
        months.append(current)
        current = current + pd.offsets.MonthBegin(1)
    return months


def _month_url(symbol: str, month: pd.Timestamp) -> str:
    return (
        f"{ARCHIVE_ROOT}/monthly/klines/{symbol}/{INTERVAL}/"
        f"{symbol}-{INTERVAL}-{month:%Y-%m}.zip"
    )


def _day_url(symbol: str, day: pd.Timestamp) -> str:
    return (
        f"{ARCHIVE_ROOT}/daily/klines/{symbol}/{INTERVAL}/"
        f"{symbol}-{INTERVAL}-{day:%Y-%m-%d}.zip"
    )


def _read_kline_archive(payload: bytes) -> DataFrame:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv_names = [
                name for name in archive.namelist() if name.lower().endswith(".csv")
            ]
            if len(csv_names) != 1:
                raise ValueError("ai_core_archive_csv_count_invalid")
            with archive.open(csv_names[0]) as stream:
                frame = pd.read_csv(
                    stream, header=None, usecols=[0, 1, 2, 4, 5, 7]
                )
    except (zipfile.BadZipFile, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"ai_core_archive_invalid:{error}") from error
    frame.columns = [
        "open_time",
        "open",
        "high",
        "close",
        "base_volume",
        "quote_volume",
    ]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open_time", "open", "high", "close", "quote_volume"])
    if frame.empty:
        raise ValueError("ai_core_archive_empty")
    if (
        (frame["open"] <= 0).any()
        or (frame["high"] <= 0).any()
        or (frame["close"] <= 0).any()
        or (frame["quote_volume"] < 0).any()
    ):
        raise ValueError("ai_core_archive_values_out_of_range")
    unit = "us" if float(frame["open_time"].max()) >= 100_000_000_000_000 else "ms"
    frame["date"] = pd.to_datetime(
        frame["open_time"].astype("int64"), unit=unit, utc=True, errors="raise"
    )
    return frame.loc[:, ["date", "open", "high", "close", "quote_volume"]].sort_values(
        "date"
    ).reset_index(drop=True)


def _download_month(
    symbol: str, month: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[list[DataFrame], list[dict[str, str]]]:
    month_url = _month_url(symbol, month)
    payload = _fetch_archive(month_url)
    if payload is not None:
        return (
            [_read_kline_archive(payload)],
            [{"url": month_url, "sha256": hashlib.sha256(payload).hexdigest()}],
        )
    first_month = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    last_month = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    if month not in (first_month, last_month):
        raise ValueError(f"ai_core_interior_archive_missing:{month:%Y-%m}")
    first_day = max(month, start.floor("D"))
    last_day_in_month = month + pd.offsets.MonthBegin(1) - pd.Timedelta(days=1)
    last_day = min(last_day_in_month, end.floor("D"))
    pieces: list[DataFrame] = []
    manifests: list[dict[str, str]] = []
    day = first_day
    while day <= last_day:
        day_url = _day_url(symbol, day)
        daily_payload = _fetch_archive(day_url)
        if daily_payload is None:
            raise ValueError(f"ai_core_boundary_archive_missing:{day:%Y-%m-%d}")
        pieces.append(_read_kline_archive(daily_payload))
        manifests.append(
            {"url": day_url, "sha256": hashlib.sha256(daily_payload).hexdigest()}
        )
        day = day + pd.Timedelta(days=1)
    return pieces, manifests


def _download_eth_4h(
    start: pd.Timestamp, end: pd.Timestamp, *, timestamp_phase_minutes: int
) -> tuple[DataFrame, dict[str, Any]]:
    pieces: list[DataFrame] = []
    archives: list[dict[str, str]] = []
    for month in _month_starts(start, end):
        month_pieces, month_archives = _download_month(SYMBOL, month, start, end)
        pieces.extend(month_pieces)
        archives.extend(month_archives)
    hourly = pd.concat(pieces, ignore_index=True)
    upper = end.floor("D") + pd.Timedelta(hours=23)
    hourly = hourly.loc[hourly["date"].between(start.floor("D"), upper)]
    hourly = hourly.sort_values("date").reset_index(drop=True)
    if hourly.empty or hourly["date"].duplicated().any():
        raise ValueError("ai_core_eth_source_empty_or_ambiguous")
    hourly = hourly.rename(columns={"quote_volume": "volume"})
    eth_4h = aggregate_hourly_to_4h(hourly)
    eth_4h["date"] = eth_4h["date"] + pd.Timedelta(
        minutes=timestamp_phase_minutes
    )
    manifest = {
        "venue": "binance_spot",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "rows_hourly": len(hourly),
        "rows_4h": len(eth_4h),
        "timestamp_phase_label_shift_minutes": timestamp_phase_minutes,
        "exact_same_window_claimed": False,
        "archives": archives,
        "source_semantics": (
            "Binance hourly OHLCV windows are label-shifted to match Nobitex's fixed "
            "timestamp phase for cross-asset sensitivity; source windows are not identical."
        ),
    }
    return eth_4h, manifest


def _load_research_registry(project_root: Path) -> tuple[Any, str]:
    path = project_root / "configs" / "research" / "ai-core-strategy-registry.json"
    raw = path.read_text(encoding="utf-8")
    result = load_registry_document(raw)
    if result.registry is None or result.reason_codes:
        raise ValueError("ai_core_research_registry_invalid")
    registry = result.registry
    identifiers = tuple(item.strategy_id for item in registry.strategies)
    expected = ("mean_reversion", "momentum_breakout", "trend_following")
    if identifiers != expected:
        raise ValueError("ai_core_research_registry_candidate_set_mismatch")
    strategy_path = project_root / "services" / "ai_core" / "strategies.py"
    strategy_hash = hashlib.sha256(strategy_path.read_bytes()).hexdigest()
    if any(item.code_sha256 != strategy_hash for item in registry.strategies):
        raise ValueError("ai_core_research_registry_code_hash_mismatch")
    return registry, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _walk_forward_folds(
    sample_count: int,
    *,
    min_train_rows: int,
    horizon: int,
    folds: int,
) -> list[tuple[int, int, int]]:
    first_test = min_train_rows + horizon
    available = sample_count - first_test
    if available < folds:
        raise ValueError("ai_core_insufficient_rows_for_walk_forward")
    result: list[tuple[int, int, int]] = []
    for index in range(folds):
        test_start = first_test + (available * index) // folds
        test_stop = first_test + (available * (index + 1)) // folds
        train_stop = test_start - horizon
        if train_stop < min_train_rows or test_stop <= test_start:
            raise ValueError("ai_core_walk_forward_fold_invalid")
        result.append((train_stop, test_start, test_stop))
    return result


def _trade_stats(returns: list[float]) -> dict[str, Any]:
    if not returns:
        return {
            "trade_count": 0,
            "win_rate": None,
            "net_expectancy": None,
            "net_compounded_return": 0.0,
            "max_abs_drawdown": 0.0,
        }
    array = np.asarray(returns, dtype=float)
    equity = np.cumprod(1.0 + array)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    return {
        "trade_count": len(returns),
        "win_rate": float(np.mean(array > 0)),
        "net_expectancy": float(np.mean(array)),
        "net_compounded_return": float(equity[-1] - 1.0),
        "max_abs_drawdown": float(abs(np.min(equity / peaks - 1.0))),
    }


def _evaluate_candidate_signals(
    test_rows: DataFrame,
    targets: np.ndarray,
    predictions: np.ndarray,
    regimes: tuple[str, ...],
    registry: Any,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    summaries: dict[str, dict[str, Any]] = {
        "trend_following": {"routed_rows": 0, "candidate_signals": 0, "net_returns": []},
        "mean_reversion": {"routed_rows": 0, "candidate_signals": 0, "net_returns": []},
        "momentum_breakout": {
            "routed_rows": 0,
            "candidate_signals": 0,
            "net_returns": [],
        },
    }
    prediction_rows: list[dict[str, Any]] = []
    next_allowed: dict[str, int] = {name: 0 for name in summaries}
    for offset, (_, row) in enumerate(test_rows.iterrows()):
        route = route_research_regime(regimes[offset], registry)
        strategy_id = route.strategy_id
        candidate_signal = 0
        accepted = False
        reason = route.reason
        if strategy_id is not None:
            summary = summaries[strategy_id]
            summary["routed_rows"] = int(summary["routed_rows"]) + 1
            signal = generate_candidate_signal(strategy_id, row.to_dict())
            candidate_signal = signal.signal
            reason = signal.reason
            if candidate_signal and predictions[offset] > ROUND_TRIP_COST:
                summary["candidate_signals"] = int(summary["candidate_signals"]) + 1
                if offset >= next_allowed[strategy_id]:
                    net_return = float(targets[offset] - ROUND_TRIP_COST)
                    cast_returns = summary["net_returns"]
                    if isinstance(cast_returns, list):
                        cast_returns.append(net_return)
                    next_allowed[strategy_id] = offset + FORWARD_HORIZON_4H
                    accepted = True
                    reason = "research_candidate_passed_ensemble_cost_filter"
                else:
                    reason = "overlapping_trade_suppressed"
        prediction_rows.append(
            {
                "date": row["date"].isoformat(),
                "hmm_regime": regimes[offset],
                "routed_strategy": strategy_id,
                "candidate_signal": candidate_signal,
                "ensemble_predicted_return": float(predictions[offset]),
                "realized_forward_return": float(targets[offset]),
                "research_trade_accepted": accepted,
                "route_or_signal_reason": reason,
                "order_permission": False,
            }
        )
    for item in summaries.values():
        raw_returns = item.pop("net_returns")
        values = raw_returns if isinstance(raw_returns, list) else []
        item.update(_trade_stats(values))
    return summaries, prediction_rows


def _evaluate_fold(
    data: DataFrame,
    regime_features: DataFrame,
    fold_index: int,
    train_stop: int,
    test_start: int,
    test_stop: int,
    registry: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = data.iloc[:train_stop]
    test = data.iloc[test_start:test_stop]
    feature_names = list(AI_FEATURE_COLUMNS)
    train_features = train.loc[:, feature_names]
    test_features = test.loc[:, feature_names]
    model_result = fit_weighted_ensemble(
        train_features,
        train["forward_return"],
        test_features,
        config=EnsembleConfig(n_estimators=48, random_seed=42 + fold_index),
    )
    actual = test["forward_return"].to_numpy(dtype=float)
    predictions = np.asarray(model_result.predictions, dtype=float)

    training_regimes = regime_features.loc[train.index, :]
    test_regimes = regime_features.loc[test.index, :]
    if len(training_regimes) < 100:
        raise ValueError("ai_core_hmm_training_rows_below_minimum")
    fit = fit_gaussian_hmm(training_regimes, max_iter=24, random_seed=42 + fold_index)
    causal_regimes, posterior = infer_hmm_regimes(
        pd.concat([training_regimes, test_regimes], axis=0), fit
    )
    labels = causal_regimes[len(training_regimes) :]
    if len(posterior) != len(training_regimes) + len(test_regimes):
        raise ValueError("ai_core_hmm_posterior_accounting_mismatch")
    strategy_summary, rows = _evaluate_candidate_signals(
        test,
        actual,
        predictions,
        labels,
        registry,
    )
    per_model_metrics = {
        name: regression_metrics(
            actual,
            np.asarray(component_predictions, dtype=float),
        )
        for name, component_predictions in model_result.per_model_predictions
    }
    return (
        {
            "fold": fold_index,
            "training_rows": len(train),
            "training_stop_exclusive": train_stop,
            "test_start": test_start,
            "test_stop_exclusive": test_stop,
            "purge_horizon_bars": FORWARD_HORIZON_4H,
            "training_end_precedes_test_by_horizon": train_stop <= test_start - FORWARD_HORIZON_4H,
            "ensemble_metrics": regression_metrics(actual, predictions),
            "component_model_metrics": per_model_metrics,
            "hmm_mean_log_likelihood": fit.mean_log_likelihood,
            "hmm_iterations": fit.iterations,
            "hmm_state_occupancy_training": dict(
                zip(
                    ("trend", "range", "high_volatility"),
                    fit.state_occupancy,
                    strict=True,
                )
            ),
            "strategy_research_diagnostics": strategy_summary,
        },
        rows,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    DataFrame(rows).to_csv(path, index=False)


def run_rebuild(input_dir: Path, output_dir: Path, commit_sha: str) -> dict[str, Any]:
    if len(commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in commit_sha
    ):
        raise ValueError("ai_core_commit_sha_invalid")
    project_root = Path(__file__).resolve().parents[1]
    registry, registry_sha = _load_research_registry(project_root)
    hourly_btc, source_report, segment_hash = stage_b._load_development(input_dir)
    minute_phases = tuple(
        sorted(pd.to_datetime(hourly_btc["date"], utc=True).dt.minute.unique().tolist())
    )
    if len(minute_phases) != 1:
        raise ValueError("ai_core_btc_timestamp_phase_not_uniform")
    btc_4h = aggregate_hourly_to_4h(hourly_btc)
    eth_4h, eth_manifest = _download_eth_4h(
        pd.Timestamp(btc_4h["date"].iloc[0]),
        pd.Timestamp(btc_4h["date"].iloc[-1]),
        timestamp_phase_minutes=int(minute_phases[0]),
    )
    features = build_ai_features(
        btc_4h,
        cross_asset={"BTCUSDT": btc_4h, "ETHUSDT": eth_4h},
    )
    forward_return = btc_4h["close"].shift(-FORWARD_HORIZON_4H) / btc_4h["close"] - 1.0
    valid = (
        features.loc[:, list(AI_FEATURE_COLUMNS)].notna().all(axis=1)
        & forward_return.replace([np.inf, -np.inf], np.nan).notna()
    )
    data = features.loc[valid].copy()
    data["forward_return"] = forward_return.loc[valid]
    regime_features = calculate_regime_features(btc_4h).loc[data.index]
    if len(data) <= MINIMUM_TRAIN_ROWS + FORWARD_HORIZON_4H + FOLD_COUNT:
        raise ValueError("ai_core_research_rows_insufficient")
    fold_specs = _walk_forward_folds(
        len(data),
        min_train_rows=MINIMUM_TRAIN_ROWS,
        horizon=FORWARD_HORIZON_4H,
        folds=FOLD_COUNT,
    )

    fold_reports: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fold_index, (train_stop, test_start, test_stop) in enumerate(fold_specs, start=1):
        report, rows = _evaluate_fold(
            data,
            regime_features,
            fold_index,
            train_stop,
            test_start,
            test_stop,
            registry,
        )
        fold_reports.append(report)
        predictions.extend(rows)

    strategy_summary: dict[str, dict[str, Any]] = {}
    for strategy_id in ("trend_following", "mean_reversion", "momentum_breakout"):
        fold_stats = [
            report["strategy_research_diagnostics"][strategy_id]
            for report in fold_reports
        ]
        strategy_summary[strategy_id] = {
            "routed_rows": sum(int(item["routed_rows"]) for item in fold_stats),
            "candidate_signals": sum(int(item["candidate_signals"]) for item in fold_stats),
            "trade_count": sum(int(item["trade_count"]) for item in fold_stats),
            "net_expectancy": (
                float(
                    np.average(
                        [
                            float(item["net_expectancy"]) * int(item["trade_count"])
                            for item in fold_stats
                            if item["net_expectancy"] is not None
                        ]
                    )
                    / sum(int(item["trade_count"]) for item in fold_stats)
                )
                if sum(int(item["trade_count"]) for item in fold_stats) > 0
                else None
            ),
            "max_abs_drawdown_max_across_folds": max(
                float(item["max_abs_drawdown"]) for item in fold_stats
            ),
        }

    report: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "run_status": "COMPLETED_RESEARCH_ONLY",
        "created_by_commit_sha": commit_sha,
        "source": {
            "coverage_workflow_run_id": 40077609244,
            "coverage_csv_sha256": source_report["csv_sha256"],
            "coverage_manifest_sha256": source_report["manifest_sha256"],
            "segment_a_rows_sha256": segment_hash,
            "symbol": "BTCUSDT",
            "source_timeframe": "1h",
            "evaluation_timeframe": "4h",
            "development_rows_hourly": len(hourly_btc),
            "development_rows_4h": len(btc_4h),
            "final_holdout_used": False,
            "reserved_final_90_day_holdout_rows": 90 * 24,
            "eth_cross_asset_manifest": eth_manifest,
        },
        "feature_engineering": {
            "schema_version": AI_FEATURE_SCHEMA_VERSION,
            "feature_count": len(AI_FEATURE_COLUMNS),
            "feature_names": list(AI_FEATURE_COLUMNS),
            "groups": [
                "returns",
                "SMA_and_EMA",
                "Bollinger_bands",
                "multi_window_RSI",
                "ATR_and_true_range",
                "realized_and_downside_volatility",
                "volume_zscores_and_turnover",
                "OHLCV_volume_profile_proxy",
                "BTC_and_ETH_cross_asset_correlation_and_beta",
            ],
            "volume_profile_limitation": (
                "Profile features are coarse bins of candle typical price weighted by OHLCV "
                "volume, not trade-level volume-at-price data."
            ),
            "cross_venue_limitation": (
                "ETHUSDT features use Binance Spot data; labels are shifted only to align "
                "the stable 30-minute source timestamp phase and do not create identical windows."
            ),
            "lookahead_policy": "trailing windows only; cross-asset joins require exact shifted labels; no backfill",
        },
        "ensemble": {
            "models": ["xgboost", "random_forest", "lightgbm"],
            "method": "equal_weighted_mean",
            "weights": {"xgboost": 1.0 / 3.0, "random_forest": 1.0 / 3.0, "lightgbm": 1.0 / 3.0},
            "target": "24-hour forward close return",
            "walk_forward_folds": fold_reports,
        },
        "strategy_registry": {
            "path": "configs/research/ai-core-strategy-registry.json",
            "revision": registry.revision,
            "sha256": registry_sha,
            "candidate_count": len(registry.strategies),
            "all_candidates_research_only": all(
                item.lifecycle.value == "RESEARCH_ONLY" and not item.owner_approved
                for item in registry.strategies
            ),
            "regime_map": {
                "trend": "trend_following",
                "range": "mean_reversion",
                "volatile": "momentum_breakout",
            },
            "runtime_approved_hmm_required": True,
            "production_registry_modified": False,
        },
        "strategy_research_diagnostics": strategy_summary,
        "sentiment": {
            "module_available": True,
            "provider_fetch_performed": False,
            "used_in_model_features": False,
            "status": "BLOCKED_PENDING_APPROVED_PERSIAN_RSS_PROVIDER_AND_CALIBRATED_MODEL",
            "reason": (
                "The RSS parser is implemented, but no provider license/retention approval "
                "or calibrated owner-reviewed NLP model was present in this research branch."
            ),
        },
        "economic_evaluation": {
            "status": "NOT_EVALUATED_RESEARCH_ONLY",
            "economic_acceptance": "NO-GO",
            "reason": "Technical metrics are diagnostics only and are not owner-approved economic evidence.",
        },
        "safety_invariants": {
            "Champion": None,
            "order_permission": False,
            "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
            "economic_acceptance": "NO-GO",
            "trading_authorization": "NO-GO",
            "model_promoted": False,
            "production_model_updated": False,
            "production_strategy_registry_updated": False,
            "orders_or_execution_api_used": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
        },
    }
    if (
        report["safety_invariants"]["Champion"] is not None
        or report["safety_invariants"]["order_permission"] is not False
    ):
        raise ValueError("ai_core_safety_invariant_violated")
    _write_json(output_dir / "ai-core-rebuild-report.json", report)
    _write_csv(output_dir / "ai-core-oos-predictions.csv", predictions)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    report = run_rebuild(args.input_dir, args.output_dir, args.commit_sha)
    print(
        json.dumps(
            {
                "commit_sha": report["created_by_commit_sha"],
                "run_status": report["run_status"],
                "timeframe": report["source"]["evaluation_timeframe"],
                "features": report["feature_engineering"]["feature_count"],
                "ensemble_models": report["ensemble"]["models"],
                "strategies": report["strategy_registry"]["candidate_count"],
                "oos_predictions": sum(
                    item["test_stop_exclusive"] - item["test_start"]
                    for item in report["ensemble"]["walk_forward_folds"]
                ),
                "economic_status": report["economic_evaluation"]["status"],
                "Champion": report["safety_invariants"]["Champion"],
                "order_permission": report["safety_invariants"]["order_permission"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
