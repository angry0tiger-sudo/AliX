"""Run 15m/30m Donchian breakout research with 1h TTM-R2 risk-only sizing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from research.ai_ensemble.hf_donchian import (
    FIXED_POSITION_FRACTION,
    TTM_MULTIPLIER_MAX,
    TTM_MULTIPLIER_MIN,
    backtest_donchian_split,
    validate_intraday_ohlcv,
)
from research.ai_ensemble.ttm_risk import (
    CONTEXT_LENGTH,
    PREDICTION_HORIZON,
    prepare_risk_target,
)
from scripts.fetch_nobitex_hf_ohlcv import (
    BINANCE_KLINES_ENDPOINT,
    INTRADAY_INTERVALS,
    PROXY_END_TIMESTAMP,
    PROXY_START_TIMESTAMP,
)
from scripts.run_ttm_r2_risk_research import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    _build_inference_runtime,
    _inverse_forecast_variance,
    _predict_context_only,
    sha256_file,
)
from scripts.run_ttm_r2_risk_research import verify_dataset as verify_hourly_dataset

TTM_REFRESH_HOURS = 4
TTM_RISK_HORIZON_HOURS = 24
TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85
PROXY_WINDOW_START_UTC = "2021-10-31T20:30:00Z"
PROXY_WINDOW_END_UTC = "2025-06-17T20:30:00Z"


def _intraday_manifest_range(
    manifest: dict[str, Any], interval: str
) -> tuple[int, int, int]:
    seconds = {"15m": 900, "30m": 1800}[interval]
    start_timestamp = manifest.get("start_timestamp")
    end_timestamp = manifest.get("end_timestamp")
    if isinstance(start_timestamp, bool) or not isinstance(start_timestamp, int):
        raise ValueError(f"hf_intraday_manifest_start_invalid:{interval}")
    if isinstance(end_timestamp, bool) or not isinstance(end_timestamp, int):
        raise ValueError(f"hf_intraday_manifest_end_invalid:{interval}")
    if start_timestamp != PROXY_START_TIMESTAMP or end_timestamp != PROXY_END_TIMESTAMP:
        raise ValueError(f"hf_proxy_manifest_range_invalid:{interval}")
    if (end_timestamp - start_timestamp) % seconds:
        raise ValueError(f"hf_intraday_manifest_interval_alignment_invalid:{interval}")
    expected_rows = ((end_timestamp - start_timestamp) // seconds) + 1
    return start_timestamp, end_timestamp, expected_rows


def _validate_intraday_manifest(
    manifest: dict[str, Any],
    required: dict[str, Any],
    interval: str,
) -> None:
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"hf_intraday_manifest_mismatch:{interval}:{key}")


def _read_intraday_frame(
    csv_path: Path,
    manifest: dict[str, Any],
    interval: str,
    *,
    start_timestamp: int,
    end_timestamp: int,
    expected_rows: int,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if len(frame) != expected_rows:
        raise ValueError(f"hf_intraday_row_count_mismatch:{interval}")
    frame = validate_intraday_ohlcv(frame, interval)
    timestamps = frame["date"]
    if int(timestamps.iloc[0].timestamp()) != start_timestamp:
        raise ValueError(f"hf_intraday_start_mismatch:{interval}")
    if int(timestamps.iloc[-1].timestamp()) != end_timestamp:
        raise ValueError(f"hf_intraday_end_mismatch:{interval}")
    if pd.Timestamp(manifest["start_utc"]) != timestamps.iloc[0]:
        raise ValueError(f"hf_intraday_manifest_start_utc_mismatch:{interval}")
    if pd.Timestamp(manifest["end_utc"]) != timestamps.iloc[-1]:
        raise ValueError(f"hf_intraday_manifest_end_utc_mismatch:{interval}")
    return frame


def _load_intraday_dataset(
    csv_path: Path, manifest_path: Path, interval: str, commit_sha: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = cast(
        dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    digest = sha256_file(csv_path)
    start_timestamp, end_timestamp, expected_rows = _intraday_manifest_range(
        manifest, interval
    )
    required = {
        "dataset_id": f"PROXY_BINANCE_{interval.upper()}",
        "dataset_role": "RESEARCH_PROXY_ONLY",
        "provider": "Binance Spot public market-data REST API",
        "endpoint": BINANCE_KLINES_ENDPOINT,
        "venue": "BINANCE_SPOT",
        "market_type": "SPOT",
        "symbol": "BTCUSDT",
        "execution_venue": "NOBITEX",
        "interval": interval,
        "native_interval": True,
        "dataset_window": "2021_11_TO_2025_06_17_PROXY",
        "source_policy": "BINANCE_PROXY_RESEARCH_ONLY",
        "proxy_not_execution_truth": True,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "requested_start_timestamp": start_timestamp,
        "requested_end_timestamp": end_timestamp,
        "row_count": expected_rows,
        "expected_candles": expected_rows,
        "missing_intervals": 0,
        "csv_sha256": digest,
        "commit_sha": commit_sha,
        "public_api_only": True,
        "market_data_only_endpoint": True,
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
    public_api_requests = manifest.get("public_api_requests")
    if (
        isinstance(public_api_requests, bool)
        or not isinstance(public_api_requests, int)
        or public_api_requests < 1
    ):
        raise ValueError(f"hf_proxy_manifest_public_api_requests_invalid:{interval}")
    _validate_intraday_manifest(manifest, required, interval)
    if digest != manifest["csv_sha256"]:
        raise ValueError(f"hf_intraday_sha256_mismatch:{interval}")
    frame = _read_intraday_frame(
        csv_path,
        manifest,
        interval,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        expected_rows=expected_rows,
    )
    return frame, manifest


def _common_split_boundaries(
    frames: dict[str, pd.DataFrame], hourly_ohlcv: pd.DataFrame
) -> tuple[pd.Timestamp, pd.Timestamp]:
    first = max(pd.Timestamp(frame["date"].iloc[0]) for frame in frames.values())
    last = min(pd.Timestamp(frame["date"].iloc[-1]) for frame in frames.values())
    if first >= last:
        raise ValueError("hf_intraday_timeframe_windows_do_not_overlap")
    span = last - first
    validation_candidate = first + span * TRAIN_FRACTION
    holdout_candidate = first + span * VALIDATION_END_FRACTION
    hourly_times = pd.to_datetime(hourly_ohlcv["date"], utc=True)
    hourly_ns = hourly_times.astype("int64").to_numpy(dtype=np.int64)
    validation_index = int(np.searchsorted(hourly_ns, validation_candidate.value))
    holdout_index = int(np.searchsorted(hourly_ns, holdout_candidate.value))
    if validation_index >= len(hourly_times) or holdout_index >= len(hourly_times):
        raise ValueError("hf_hourly_split_alignment_out_of_range")
    validation_start = pd.Timestamp(hourly_times.iloc[validation_index])
    holdout_start = pd.Timestamp(hourly_times.iloc[holdout_index])
    if not first < validation_start < holdout_start < last:
        raise ValueError("hf_chronological_split_boundaries_invalid")
    return validation_start, holdout_start


def _training_reference_variance(return_variance: np.ndarray, train_end: int) -> float:
    if train_end < TTM_RISK_HORIZON_HOURS + CONTEXT_LENGTH:
        raise ValueError("hf_ttm_training_prefix_too_short")
    training = np.asarray(return_variance[:train_end], dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(training)))
    rolling_24h = (
        cumulative[TTM_RISK_HORIZON_HOURS:] - cumulative[:-TTM_RISK_HORIZON_HOURS]
    )
    positive = rolling_24h[np.isfinite(rolling_24h) & (rolling_24h > 0.0)]
    if positive.size < 30:
        raise ValueError("hf_ttm_training_reference_has_too_few_windows")
    reference = float(np.median(positive))
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("hf_ttm_training_reference_invalid")
    return reference


def _build_hourly_ttm_risk_forecasts(
    hourly_ohlcv: pd.DataFrame, validation_start: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = prepare_risk_target(hourly_ohlcv)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    train_end = int(np.searchsorted(timestamps.astype("int64"), validation_start.value))
    return_variance = frame["return_variance"].to_numpy(dtype=np.float64)
    reference = _training_reference_variance(return_variance, train_end)
    first_origin = max(CONTEXT_LENGTH - 1, train_end - 1)
    origins = list(
        range(first_origin, len(frame) - PREDICTION_HORIZON, TTM_REFRESH_HOURS)
    )
    if len(origins) < 30:
        raise ValueError("hf_ttm_forecast_origins_insufficient")

    runtime = _build_inference_runtime(frame, train_end)
    predictions = _predict_context_only(runtime["model"], runtime["dataset"], origins)
    forecast_variance, _, floor_fraction = _inverse_forecast_variance(
        predictions,
        runtime["preprocessor"],
        len(origins),
        float(runtime["minimum_log1p_variance_target"]),
    )
    multipliers = np.clip(
        np.sqrt(reference / forecast_variance),
        TTM_MULTIPLIER_MIN,
        TTM_MULTIPLIER_MAX,
    )
    available_at = [timestamps.iloc[origin + 1] for origin in origins]
    risk = pd.DataFrame(
        {
            "available_at": available_at,
            "forecast_variance_24h": forecast_variance,
            "position_multiplier": multipliers,
            "target_floor_fraction": floor_fraction,
        }
    )
    metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weights_sha256": MODEL_WEIGHTS_SHA256,
        "mode": "ZERO_SHOT_FROZEN_WEIGHTS",
        "fine_tuned": False,
        "risk_horizon_hours": TTM_RISK_HORIZON_HOURS,
        "refresh_cadence_hours": TTM_REFRESH_HOURS,
        "training_prefix_end_exclusive": timestamps.iloc[train_end - 1].isoformat(),
        "training_reference_median_24h_variance": reference,
        "forecast_count": len(risk),
        "minimum_multiplier": float(np.min(multipliers)),
        "median_multiplier": float(np.median(multipliers)),
        "maximum_multiplier": float(np.max(multipliers)),
        "target_floor_applied_fraction": float(np.mean(floor_fraction)),
    }
    return risk, metadata


def _timeframe_result(
    frame: pd.DataFrame,
    risk: pd.DataFrame,
    interval: str,
    validation_start: pd.Timestamp,
    holdout_start: pd.Timestamp,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    timestamps = pd.to_datetime(frame["date"], utc=True)
    validation_index = int(
        np.searchsorted(timestamps.astype("int64"), validation_start.value)
    )
    holdout_index = int(
        np.searchsorted(timestamps.astype("int64"), holdout_start.value)
    )
    if (
        validation_index == 0
        or holdout_index <= validation_index
        or holdout_index >= len(frame)
    ):
        raise ValueError(f"hf_split_index_invalid:{interval}")

    validation = backtest_donchian_split(
        frame,
        risk,
        timeframe=interval,
        start_index=validation_index,
        end_index=holdout_index,
    )
    holdout = backtest_donchian_split(
        frame,
        risk,
        timeframe=interval,
        start_index=holdout_index,
        end_index=len(frame),
    )
    summary: dict[str, Any] = {
        "dataset": {
            "interval": interval,
            "row_count": len(frame),
            "start_utc": timestamps.iloc[0].isoformat(),
            "end_utc": timestamps.iloc[-1].isoformat(),
            "validation_start_utc": validation_start.isoformat(),
            "holdout_start_utc": holdout_start.isoformat(),
        },
        "validation": {
            "fixed_size_baseline": validation["fixed_size_baseline"],
            "ttm_risk_managed": validation["ttm_risk_managed"],
            "same_trade_count": validation["same_trade_count"],
        },
        "final_holdout": {
            "fixed_size_baseline": holdout["fixed_size_baseline"],
            "ttm_risk_managed": holdout["ttm_risk_managed"],
            "same_trade_count": holdout["same_trade_count"],
        },
    }
    managed = holdout["ttm_risk_managed"]
    passed = bool(
        managed["net_profit_factor"] is not None
        and float(managed["net_profit_factor"]) >= 1.1
        and float(managed["max_drawdown"]) < 0.15
        and float(managed["trades_per_year"]) > 200.0
    )
    summary["acceptance"] = {
        "net_pf_at_least_1_1": bool(
            managed["net_profit_factor"] is not None
            and float(managed["net_profit_factor"]) >= 1.1
        ),
        "max_drawdown_below_15_percent": float(managed["max_drawdown"]) < 0.15,
        "more_than_200_trades_per_year": float(managed["trades_per_year"]) > 200.0,
        "all_holdout_gates_passed": passed,
    }
    return summary, validation["trades"] + holdout["trades"]


def _write_step_summary(report: dict[str, Any]) -> None:
    lines = [
        "# High-Frequency Donchian + TTM-R2 Binance Proxy Research",
        f"Commit: {report['code_commit_sha']}",
        f"Requested proxy window: {report['dataset_window']['requested_start_utc']} to {report['dataset_window']['requested_end_utc']}",
        f"Loaded proxy window: {report['dataset_window']['start_utc']} to {report['dataset_window']['end_utc']}",
        "15m/30m OHLCV is Binance Spot proxy data, not Nobitex quotes or fills.",
        "Hourly TTM-R2 risk context remains Nobitex 1h; this mixed-venue analysis is research-only.",
        "No resampling or interpolation is used; live execution remains disabled on Nobitex.",
        "",
        "## Strategy protocol",
        "| Parameter | Value |",
        "|---|---:|",
        "| Donchian entry / opposite exit | 20 prior candles |",
        "| ATR | Wilder-style 14 |",
        "| Stop loss | 2 ATR |",
        "| Take profit | 4 ATR |",
        "| Execution | Signal at close, entry at next open |",
        "| Intrabar collision | Stop loss has priority |",
        "| Max drawdown | Candle-close mark-to-market while in position |",
        "| Round-trip cost | 50 bps, deducted once per completed trade |",
        f"| Fixed notional baseline | {FIXED_POSITION_FRACTION:.2%} |",
        f"| TTM multiplier bounds | {TTM_MULTIPLIER_MIN:.2f}x to {TTM_MULTIPLIER_MAX:.2f}x |",
        "| TTM input | Nobitex 1h data; next-24h variance forecast |",
        f"| TTM refresh | Every {TTM_REFRESH_HOURS} hours; latest forecast available as-of entry |",
        "",
        "## Final holdout metrics",
        "| Timeframe | Sizing | Net PF | Max DD | Win Rate | Trades | Trades / year | Net Return |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for timeframe, result in report["timeframes"].items():
        for label, key in (
            ("Fixed", "fixed_size_baseline"),
            ("TTM", "ttm_risk_managed"),
        ):
            metrics = result["final_holdout"][key]
            pf = (
                "N/A"
                if metrics["net_profit_factor"] is None
                else f"{metrics['net_profit_factor']:.4f}"
            )
            win = "N/A" if metrics["win_rate"] is None else f"{metrics['win_rate']:.2%}"
            lines.append(
                f"| {timeframe} | {label} | {pf} | {metrics['max_drawdown']:.2%} | "
                f"{win} | {metrics['trade_count']} | {metrics['trades_per_year']:.1f} | "
                f"{metrics['net_return']:.2%} |"
            )
    lines.extend(
        [
            "",
            f"**Decision: {report['verdict']}**",
            "",
            f"TTM model: {report['ttm_risk']['model_id']} @ {report['ttm_risk']['model_revision']}",
            f"Forecasts: {report['ttm_risk']['forecast_count']}; train-only reference variance: "
            f"{report['ttm_risk']['training_reference_median_24h_variance']:.10g}",
            "",
            "Safety: Champion=null; order_permission=false; trading_authorization=NO-GO; "
            "private_api_calls=0; real_order_attempts=0; paper_orders=0; "
            "position_multiplier_applied_to_runtime=false.",
        ]
    )
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ohlcv-15m", type=Path, required=True)
    parser.add_argument("--manifest-15m", type=Path, required=True)
    parser.add_argument("--ohlcv-30m", type=Path, required=True)
    parser.add_argument("--manifest-30m", type=Path, required=True)
    parser.add_argument("--hourly-ohlcv", type=Path, required=True)
    parser.add_argument("--hourly-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    return parser


def _load_research_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, Any]],
    pd.DataFrame,
    dict[str, Any],
    pd.Timestamp,
    pd.Timestamp,
]:
    loaded: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict[str, Any]] = {}
    file_inputs = {
        "15m": (args.ohlcv_15m, args.manifest_15m),
        "30m": (args.ohlcv_30m, args.manifest_30m),
    }
    for interval, (csv_path, manifest_path) in file_inputs.items():
        loaded[interval], manifests[interval] = _load_intraday_dataset(
            csv_path, manifest_path, interval, args.commit_sha
        )
    hourly_ohlcv, hourly_manifest = verify_hourly_dataset(
        args.hourly_ohlcv, args.hourly_manifest, args.commit_sha
    )
    validation_start, holdout_start = _common_split_boundaries(loaded, hourly_ohlcv)
    return (
        loaded,
        manifests,
        hourly_ohlcv,
        hourly_manifest,
        validation_start,
        holdout_start,
    )


def _evaluate_timeframes(
    loaded: dict[str, pd.DataFrame],
    risk_forecasts: pd.DataFrame,
    validation_start: pd.Timestamp,
    holdout_start: pd.Timestamp,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: dict[str, Any] = {}
    trade_rows: list[dict[str, Any]] = []
    for interval in INTRADAY_INTERVALS:
        result, trades = _timeframe_result(
            loaded[interval],
            risk_forecasts,
            interval,
            validation_start,
            holdout_start,
        )
        results[interval] = result
        trade_rows.extend(
            [
                {
                    **trade,
                    "split": (
                        "validation"
                        if pd.Timestamp(trade["entry_time"]) < holdout_start
                        else "holdout"
                    ),
                }
                for trade in trades
            ]
        )
    return results, trade_rows


def _assemble_report(
    args: argparse.Namespace,
    manifests: dict[str, dict[str, Any]],
    hourly_manifest: dict[str, Any],
    validation_start: pd.Timestamp,
    holdout_start: pd.Timestamp,
    risk_metadata: dict[str, Any],
    timeframe_results: dict[str, Any],
) -> dict[str, Any]:
    passed = any(
        result["acceptance"]["all_holdout_gates_passed"]
        for result in timeframe_results.values()
    )
    verdict = (
        "PROXY_EDGE_FOUND_REQUIRES_NOBITEX_PAPER_VALIDATION"
        if passed
        else "NO_STATISTICAL_EDGE_FOUND_IN_HIGH_FREQUENCY_PROXY_DATA"
    )
    safety = {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "ALIX_DRY_RUN": True,
        "ALIX_ALLOW_LIVE_TRADING": False,
        "public_market_data_calls": sum(
            int(manifests[interval]["public_api_requests"])
            for interval in INTRADAY_INTERVALS
        ),
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "paper_orders": 0,
        "position_multiplier_applied_to_runtime": False,
        "offline_position_sizing_simulated": True,
        "model_generates_entry_signals": False,
    }
    if safety["Champion"] is not None or safety["trading_authorization"] != "NO-GO":
        raise ValueError("hf_safety_invariant_violated")

    return {
        "experiment": "Binance Spot 15m/30m proxy Donchian-20 ATR(14) with Nobitex 1h TTM-R2 risk-only sizing",
        "code_commit_sha": args.commit_sha,
        "dataset_window": {
            "start_utc": max(
                manifests[interval]["start_utc"] for interval in INTRADAY_INTERVALS
            ),
            "end_utc": min(
                manifests[interval]["end_utc"] for interval in INTRADAY_INTERVALS
            ),
            "requested_start_utc": PROXY_WINDOW_START_UTC,
            "requested_end_utc": PROXY_WINDOW_END_UTC,
            "intraday_data_role": "BINANCE_SPOT_PROXY_RESEARCH_ONLY",
            "intraday_source_policy": "BINANCE_PROXY_RESEARCH_ONLY",
            "execution_venue": "NOBITEX",
            "hourly_risk_context_venue": "NOBITEX_1H",
            "note": (
                "15m/30m inputs are Binance Spot proxy candles, not Nobitex prices or fills. "
                "The TTM-R2 risk context is Nobitex 1h history. This mixed-venue setup is "
                "research-only and does not establish execution equivalence."
            ),
            "source_policy": "BINANCE_PROXY_RESEARCH_ONLY",
            "hourly_context_sha256": hourly_manifest["csv_sha256"],
            "datasets": {
                interval: {
                    "dataset_id": manifests[interval]["dataset_id"],
                    "provider": manifests[interval]["provider"],
                    "venue": manifests[interval]["venue"],
                    "interval": manifests[interval]["interval"],
                    "public_api_requests": manifests[interval]["public_api_requests"],
                    "row_count": manifests[interval]["row_count"],
                    "csv_sha256": manifests[interval]["csv_sha256"],
                    "start_utc": manifests[interval]["start_utc"],
                    "end_utc": manifests[interval]["end_utc"],
                }
                for interval in INTRADAY_INTERVALS
            },
        },
        "strategy": {
            "timeframes": list(INTRADAY_INTERVALS),
            "direction": "LONG_ONLY",
            "entry": "close above highest high of previous 20 completed candles",
            "opposite_exit": "close below lowest low of previous 20 completed candles; exit at next open",
            "atr_window": 14,
            "stop_loss_atr": 2.0,
            "take_profit_atr": 4.0,
            "execution": "signal at candle close; entry at next candle open",
            "intrabar_collision_policy": "STOP_LOSS_FIRST",
            "max_drawdown_method": "CANDLE_CLOSE_MARK_TO_MARKET_WHILE_POSITION_OPEN_ACTUAL_EXIT_FILL_ON_EXIT_CANDLE",
            "round_trip_cost_bps": 50,
            "fixed_position_fraction": FIXED_POSITION_FRACTION,
            "ttm_multiplier_bounds": [TTM_MULTIPLIER_MIN, TTM_MULTIPLIER_MAX],
            "no_leverage": True,
            "risk_only_overlay": True,
            "intraday_research_venue": "BINANCE_SPOT_PROXY",
            "ttm_risk_context_venue": "NOBITEX_1H",
            "intended_execution_venue": "NOBITEX",
            "proxy_execution_equivalence_assumed": False,
        },
        "chronological_protocol": {
            "training_fraction": TRAIN_FRACTION,
            "validation_end_fraction": VALIDATION_END_FRACTION,
            "validation_start_utc": validation_start.isoformat(),
            "holdout_start_utc": holdout_start.isoformat(),
            "entry_signals_and_metrics_are_split_isolated": True,
            "ttm_scaler_and_risk_reference_fit_only_on_training_prefix": True,
        },
        "ttm_risk": risk_metadata,
        "timeframes": timeframe_results,
        "verdict": verdict,
        "acceptance": {
            "net_profit_factor_minimum": 1.1,
            "max_drawdown_strictly_below": 0.15,
            "minimum_trades_per_year_strictly_above": 200.0,
            "any_timeframe_passed": passed,
            "dataset_scope": "BINANCE_SPOT_PROXY_RESEARCH_ONLY",
            "nobitex_paper_validation_required_before_promotion": True,
        },
        "safety": safety,
    }


def _write_report_artifacts(
    output_dir: Path,
    report: dict[str, Any],
    trade_rows: list[dict[str, Any]],
    risk_forecasts: pd.DataFrame,
) -> None:
    pd.DataFrame(trade_rows).to_csv(output_dir / "nobitex-hf-trades.csv", index=False)
    (output_dir / "nobitex-hf-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    risk_forecasts.to_csv(output_dir / "ttm-risk-forecasts-hourly.csv", index=False)
    _write_step_summary(report)


def main() -> int:
    args = build_parser().parse_args()
    if len(args.commit_sha) != 40:
        raise ValueError("hf_commit_sha_must_be_full_git_sha")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (
        loaded,
        manifests,
        hourly_ohlcv,
        hourly_manifest,
        validation_start,
        holdout_start,
    ) = _load_research_inputs(args)
    risk_forecasts, risk_metadata = _build_hourly_ttm_risk_forecasts(
        hourly_ohlcv, validation_start
    )
    timeframe_results, trade_rows = _evaluate_timeframes(
        loaded, risk_forecasts, validation_start, holdout_start
    )
    report = _assemble_report(
        args,
        manifests,
        hourly_manifest,
        validation_start,
        holdout_start,
        risk_metadata,
        timeframe_results,
    )
    _write_report_artifacts(args.output_dir, report, trade_rows, risk_forecasts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
