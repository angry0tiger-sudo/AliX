"""Run the V4-AQ Donchian + external Funding/OI LightGBM experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from scripts import run_ml_core_real_data_baseline
from scripts import run_ml_core_v4ap_diagnostic_sweep as v4ap
from scripts.fetch_binance_public_ohlcv import SCHEMA_VERSION as SPOT_SCHEMA_VERSION
from services.meta_labeling import v4x
from services.meta_labeling.v4aq import (
    V4AQ_META_FEATURE_COLUMNS,
    build_v4aq_meta_features,
    fit_v4aq_lightgbm_meta_filter,
    predict_v4aq_meta_probabilities,
)
from services.ml_core.adaptive_targets import (
    ATR_PERIOD,
    apply_adaptive_triple_barrier,
    calculate_atr,
)

REPORT_SCHEMA_VERSION = "alix.ml-core.v4aq-ai6.v1"
SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"
REQUESTED_DAYS = 1825
TRAIN_FRACTION = 0.70
FORWARD_END_FRACTION = 0.85
OOF_FOLD_COUNT = 5
HORIZON_BARS = 6
BALANCED_EXIT = {
    "name": "BALANCED",
    "stop_loss_atr": 1.5,
    "take_profit_atr": 3.0,
    "trailing_enabled": False,
    "trailing_activation_atr": None,
    "trailing_step_atr": None,
}
FEE_BPS_PER_SIDE = 10.0
SLIPPAGE_BPS_PER_SIDE = 5.0
ROUND_TRIP_COST_BPS = 30.0
MINIMUM_TRADE_COUNT = 50
MINIMUM_WIN_RATE = 0.40
MINIMUM_NET_PROFIT_FACTOR = 1.10


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate_sha(commit_sha: str) -> None:
    if len(commit_sha) != 40 or any(c not in "0123456789abcdef" for c in commit_sha):
        raise ValueError("v4aq_commit_sha_invalid")


def _read_symbol_manifest(
    external_dir: Path,
    symbol: str,
    commit_sha: str,
) -> dict[str, Any]:
    stem = f"binance-{symbol.lower()}"
    path = external_dir / f"{stem}-manifest.json"
    if not path.is_file():
        raise ValueError(f"v4aq_external_manifest_missing:{symbol}")
    raw_manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise ValueError(f"v4aq_external_manifest_schema_invalid:{symbol}")
    manifest: dict[str, Any] = raw_manifest
    expected = (
        manifest.get("schema_version") == "alix.binance.usdm-external-market.v1",
        manifest.get("commit_sha") == commit_sha,
        manifest.get("symbol") == symbol,
        manifest.get("requested_days") == REQUESTED_DAYS,
        manifest.get("public_api_only") is True,
        manifest.get("credentials_used") is False,
        manifest.get("api_key_header_sent") is False,
        manifest.get("private_api_calls") == 0,
        manifest.get("exchange_order_calls") == 0,
        manifest.get("Champion") is None,
        manifest.get("order_permission") is False,
        manifest.get("trading_authorization") == "NO-GO",
    )
    if not all(expected):
        raise ValueError(f"v4aq_external_manifest_contract_invalid:{symbol}")
    _verify_manifested_csvs(external_dir, stem, manifest, symbol)
    return manifest


def _verify_manifested_csvs(
    external_dir: Path,
    stem: str,
    manifest: dict[str, Any],
    symbol: str,
) -> None:
    sections = (
        ("funding", "funding_csv", "funding_csv_sha256"),
        ("open_interest", "open_interest_csv", "open_interest_csv_sha256"),
    )
    for section, filename_key, digest_key in sections:
        if not isinstance(manifest.get(section), dict):
            raise ValueError(
                f"v4aq_external_manifest_section_missing:{symbol}:{section}"
            )
        path = external_dir / str(manifest.get(filename_key, ""))
        if not path.is_file() or _sha256(path) != manifest.get(digest_key):
            raise ValueError(f"v4aq_external_csv_hash_mismatch:{symbol}:{section}")


def _external_coverage(
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for symbol, manifest in manifests.items():
        funding = dict(manifest["funding"])
        oi = dict(manifest["open_interest"])
        coverage[symbol] = {
            "funding": funding,
            "open_interest": oi,
            "full_requested_history": (
                funding.get("full_requested_history_available") is True
                and oi.get("full_requested_history_available") is True
            ),
        }
    return coverage


def _coverage_is_complete(coverage: dict[str, Any]) -> bool:
    selected = coverage.get(SYMBOL)
    return isinstance(selected, dict) and selected.get("full_requested_history") is True


def _read_external_frames(
    external_dir: Path,
    symbol: str,
) -> tuple[DataFrame, DataFrame]:
    stem = f"binance-{symbol.lower()}"
    return (
        pd.read_csv(external_dir / f"{stem}-funding.csv"),
        pd.read_csv(external_dir / f"{stem}-open-interest.csv"),
    )


def _load_spot_bars(
    input_dir: Path,
    commit_sha: str,
) -> tuple[DataFrame, dict[str, Any], dict[str, Any]]:
    manifest_path = input_dir / f"binance-{SYMBOL.lower()}-1m.manifest.json"
    if not manifest_path.is_file():
        raise ValueError("v4aq_spot_manifest_missing")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != SPOT_SCHEMA_VERSION:
        raise ValueError("v4aq_spot_manifest_schema_invalid")
    bars_1m, provenance = v4ap._load_market_data(
        input_dir,
        symbol=SYMBOL,
        commit_sha=commit_sha,
        start_utc=pd.Timestamp(source_manifest["start_utc"]),
        end_utc=pd.Timestamp(source_manifest["end_utc"]),
    )
    bars = run_ml_core_real_data_baseline.resample_ohlcv(
        bars_1m, TIMEFRAME, allow_exchange_gaps=True
    )
    from scripts import run_ml_core_v4af_meta_final as v4af

    bars, gap_diagnostics = v4af._trim_to_latest_contiguous_segment(bars, TIMEFRAME)
    bars = bars.reset_index(drop=True)
    if len(bars) < 2_000:
        raise ValueError("v4aq_contiguous_spot_history_insufficient")
    return bars, provenance, gap_diagnostics


def _training_label_frame(
    bars: DataFrame,
    meta_features: DataFrame,
    primary_signal: Series,
    train_end: int,
) -> DataFrame:
    train_bars = bars.iloc[:train_end].reset_index(drop=True)
    train_signal = primary_signal.iloc[:train_end].reset_index(drop=True)
    oof_events = v4x.generate_oof_primary_events(
        train_signal,
        start_index=21,
        stop_index=train_end,
        fold_count=OOF_FOLD_COUNT,
    )
    atr = calculate_atr(train_bars, period=ATR_PERIOD)
    barrier_labels = apply_adaptive_triple_barrier(
        train_bars,
        atr_values=atr,
        atr_period=ATR_PERIOD,
        horizon_candles=HORIZON_BARS,
        take_profit_multiplier=3.0,
        stop_loss_multiplier=1.5,
        entry_reference="NEXT_BAR_OPEN",
        entry_slippage_bps=SLIPPAGE_BPS_PER_SIDE,
    )
    costed = v4x.build_costed_meta_labels(
        train_bars,
        oof_events,
        barrier_labels,
        fee_bps_per_side=FEE_BPS_PER_SIDE,
        slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
        entry_slippage_bps=SLIPPAGE_BPS_PER_SIDE,
    )
    indices = costed["event_index"].to_numpy(dtype=int)
    selected = meta_features.iloc[indices].reset_index(drop=True)
    complete = selected.notna().all(axis=1)
    complete &= np.isfinite(selected.to_numpy(dtype=float)).all(axis=1)
    costed = costed.loc[complete].reset_index(drop=True)
    selected = selected.loc[complete].reset_index(drop=True)
    selected.insert(0, "event_index", costed["event_index"].to_numpy(dtype=int))
    training = costed.merge(
        selected,
        on="event_index",
        how="inner",
        validate="one_to_one",
    ).sort_values("event_index")
    if len(training) != len(costed) or training.empty:
        raise ValueError("v4aq_meta_training_alignment_invalid")
    return training.reset_index(drop=True)


def _fit_meta_model(
    training: DataFrame,
    train_end: int,
) -> tuple[Any, dict[str, Any]]:
    fit_rows, calibration_rows = v4x.split_meta_fit_calibration(
        training,
        outer_train_end_index=train_end,
        calibration_fraction=0.15,
        minimum_fit_rows=40,
        minimum_class_rows=8,
    )
    model, diagnostics = fit_v4aq_lightgbm_meta_filter(
        fit_rows.loc[:, list(V4AQ_META_FEATURE_COLUMNS)],
        fit_rows["meta_label"],
        calibration_rows.loc[:, list(V4AQ_META_FEATURE_COLUMNS)],
        target_accept_fraction=0.50,
    )
    return model, diagnostics


def _forward_event_probabilities(
    features: DataFrame,
    primary_signal: Series,
    model: Any,
    train_end: int,
    forward_end: int,
) -> tuple[list[int], dict[int, float], int]:
    last_eligible = forward_end - HORIZON_BARS - 2
    indices = [
        index
        for index in range(train_end, last_eligible + 1)
        if bool(primary_signal.iloc[index])
    ]
    if not indices:
        raise ValueError("v4aq_forward_primary_events_empty")
    selected = features.iloc[indices].reset_index(drop=True)
    complete = selected.notna().all(axis=1)
    complete &= np.isfinite(selected.to_numpy(dtype=float)).all(axis=1)
    valid_indices = [index for index, valid in zip(indices, complete.tolist()) if valid]
    if not valid_indices:
        raise ValueError("v4aq_forward_external_features_unavailable")
    probabilities = predict_v4aq_meta_probabilities(
        model,
        features.iloc[valid_indices].reset_index(drop=True),
    )
    return (
        indices,
        {index: float(value) for index, value in zip(valid_indices, probabilities)},
        len(indices) - len(valid_indices),
    )


def _validation_signal(bars: DataFrame, indices: list[int]) -> Series:
    signal = Series(False, index=bars.index, dtype=bool, name="primary_buy_signal")
    if indices:
        signal.iloc[indices] = True
    return signal


def _run_candidate(
    bars: DataFrame,
    signal: Series,
    *,
    raw_signal_count: int,
    candidate_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics = {
        "primary_signal_count": raw_signal_count,
        "lookahead_used": False,
        "primary_rule": "CLOSE_ABOVE_PRIOR_20_COMPLETED_BAR_HIGHS",
    }
    result, trades = v4ap._run_candidate(
        bars,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        strategy="DONCHIAN20_CLOSE_BREAKOUT",
        geometry=BALANCED_EXIT,
        signal=signal,
        strategy_diagnostics=diagnostics,
        execution=v4ap._execution_config(),
    )
    result["candidate_label"] = candidate_label
    return result, trades


def _run_economic_comparison(
    bars: DataFrame,
    features: DataFrame,
    primary_signal: Series,
    model: Any,
    model_diagnostics: dict[str, Any],
    train_end: int,
    forward_end: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    events, probabilities, missing_features = _forward_event_probabilities(
        features, primary_signal, model, train_end, forward_end
    )
    threshold = float(model_diagnostics["selected_threshold"])
    accepted = [
        index
        for index, probability in probabilities.items()
        if probability >= threshold
    ]
    control_result, control_trades = _run_candidate(
        bars,
        _validation_signal(bars, events),
        raw_signal_count=len(events),
        candidate_label="UNFILTERED_CONTROL",
    )
    meta_result, meta_trades = _run_candidate(
        bars,
        _validation_signal(bars, accepted),
        raw_signal_count=len(events),
        candidate_label="META_FILTERED",
    )
    decisions = [
        {
            "event_index": index,
            "signal_timestamp": pd.Timestamp(bars["date"].iloc[index]).isoformat(),
            "meta_probability": probabilities.get(index),
            "selected_threshold": threshold,
            "decision": (
                "ACCEPT"
                if index in accepted
                else "REJECT_NO_POINT_IN_TIME_FEATURES"
                if index not in probabilities
                else "REJECT_LOW_META_PROBABILITY"
            ),
            "split_role": "FORWARD_VALIDATION",
            "executable_signal": "BUY" if index in accepted else "NO_ENTRY",
        }
        for index in events
    ]
    pf = meta_result.get("net_profit_factor")
    return (
        {
            "control": control_result,
            "meta_filtered": meta_result,
            "forward_primary_event_count": len(events),
            "meta_feature_missing_event_count": missing_features,
            "meta_accepted_event_count": len(accepted),
            "meta_rejected_event_count": len(events) - len(accepted),
            "model_threshold": threshold,
            "net_pf_gate": {
                "minimum_net_profit_factor": MINIMUM_NET_PROFIT_FACTOR,
                "passed": pf is not None and float(pf) >= MINIMUM_NET_PROFIT_FACTOR,
            },
            "full_statistical_acceptance": (
                "NOT_EVALUATED_DSR_NOT_PREREGISTERED_FOR_SINGLE_CELL"
            ),
        },
        [*control_trades, *meta_trades],
        decisions,
    )


def _candidate_acceptance(metrics: dict[str, Any]) -> dict[str, Any]:
    count = int(metrics.get("trade_count", 0))
    win = float(metrics.get("win_rate", 0.0))
    raw_pf = metrics.get("net_profit_factor")
    net_pf = None if raw_pf is None else float(raw_pf)
    return {
        "trade_count": count,
        "win_rate": win,
        "net_profit_factor": net_pf,
        "trade_count_passed": count >= 50,
        "win_rate_passed": win >= 0.40,
        "net_profit_factor_passed": net_pf is not None and net_pf >= 1.10,
        "dsr_z": None,
        "dsr_passed": False,
        "statistical_edge_accepted": False,
    }


def _build_blocked_report(
    commit_sha: str,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "BLOCKED_INSUFFICIENT_EXTERNAL_HISTORY",
        "commit_sha": commit_sha,
        "experiment": {
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "strategy": "DONCHIAN20_CLOSE_BREAKOUT",
            "exit": "BALANCED",
            "requested_days": REQUESTED_DAYS,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        },
        "external_data_coverage": coverage,
        "model": {
            "family": "LightGBM binary meta-filter",
            "fit_performed": False,
            "feature_columns": list(V4AQ_META_FEATURE_COLUMNS),
            "skip_reason": "binance_public_open_interest_history_limited_to_latest_30_days",
        },
        "economic_results": None,
        "economic_acceptance": "NOT_EVALUATED",
        "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "limitations": [
            "No five-year Binance Open Interest History is available from the documented public endpoint.",
            "The LightGBM fit and economic replay were not run on incomplete external-feature history.",
            "No missing OI observations were interpolated, backfilled or synthesized.",
        ],
    }


def _build_completed_report(
    commit_sha: str,
    spot_provenance: dict[str, Any],
    gap_diagnostics: dict[str, Any],
    coverage: dict[str, Any],
    model_diagnostics: dict[str, Any],
    comparison: dict[str, Any],
    training: DataFrame,
    external_diagnostics: dict[str, Any],
    signal_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "COMPLETED_RESEARCH_EVALUATION",
        "commit_sha": commit_sha,
        "experiment": {
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "strategy": "DONCHIAN20_CLOSE_BREAKOUT",
            "exit": "BALANCED",
            "requested_days": REQUESTED_DAYS,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "final_15_percent_untouched": True,
        },
        "spot_provenance": spot_provenance,
        "gap_diagnostics": gap_diagnostics,
        "external_data_coverage": coverage,
        "model": {
            "family": "LightGBM binary meta-filter",
            "fit_performed": True,
            "fit_diagnostics": model_diagnostics,
            "feature_columns": list(V4AQ_META_FEATURE_COLUMNS),
            "training_rows": int(len(training)),
        },
        "external_feature_diagnostics": external_diagnostics,
        "primary_signal_diagnostics": signal_diagnostics,
        "economic_results": comparison,
        "meta_acceptance": _candidate_acceptance(comparison["meta_filtered"]),
        "economic_acceptance": "NO-GO",
        "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "limitations": [
            "This single-cell forward-validation comparison is not a preregistered multi-trial promotion test.",
            "DSR is not evaluated for this single-cell follow-up; Net PF alone cannot establish statistical edge.",
        ],
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _render_summary(report: dict[str, Any]) -> str:
    lines = [
        "# AliX V4-AQ Phase AI-6",
        "",
        f"- Commit: {report['commit_sha']}",
        f"- Status: **{report['status']}**",
        f"- Candidate: {SYMBOL} / {TIMEFRAME} / DONCHIAN20 / BALANCED",
        f"- Costs: {ROUND_TRIP_COST_BPS:.2f} bps round trip",
        "",
    ]
    if report["economic_results"] is None:
        lines.extend(
            [
                "## External data coverage",
                "",
                "| Asset | Funding span days | OI span days | OI rows | Coverage status |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for symbol, item in report["external_data_coverage"].items():
            funding = item["funding"]
            oi = item["open_interest"]
            lines.append(
                f"| {symbol} | {funding['observed_span_days']:.3f} | "
                f"{oi['observed_span_days']:.3f} | {oi['row_count']} | "
                f"{oi['coverage_status']} |"
            )
        lines.extend(
            [
                "",
                "LightGBM fit: NOT RUN. External data does not cover the full "
                "1825-day training window. No imputation or short-window edge claim is allowed.",
            ]
        )
    else:
        control = report["economic_results"]["control"]
        meta = report["economic_results"]["meta_filtered"]
        lines.extend(
            [
                "## Forward-validation comparison",
                "",
                "| Mode | Trades | Win % | Net PF | Net PnL USD |",
                "|---|---:|---:|---:|---:|",
                f"| Control | {control['trade_count']} | {control['win_rate']:.2%} | "
                f"{control['net_profit_factor']} | {control['net_pnl_usd']:.2f} |",
                f"| Meta | {meta['trade_count']} | {meta['win_rate']:.2%} | "
                f"{meta['net_profit_factor']} | {meta['net_pnl_usd']:.2f} |",
                "",
                f"- Net PF >= 1.10: {report['economic_results']['net_pf_gate']['passed']}",
                "- DSR: NOT EVALUATED",
                "- Promotion: NO-GO",
            ]
        )
    lines.extend(
        [
            "",
            "Champion=null | order_permission=false | trading_authorization=NO-GO | private_api_calls=0",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_run_request(commit_sha: str) -> None:
    _validate_sha(commit_sha)
    if ROUND_TRIP_COST_BPS != v4ap.ROUND_TRIP_COST_BPS:
        raise ValueError("v4aq_v4ap_cost_contract_mismatch")


def _run_complete_experiment(
    input_dir: Path,
    external_dir: Path,
    commit_sha: str,
    coverage: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    bars, spot_provenance, gap_diagnostics = _load_spot_bars(input_dir, commit_sha)
    funding, oi = _read_external_frames(external_dir, SYMBOL)
    from services.ml_core.external_market_features import build_external_market_features

    external_features, external_diagnostics = build_external_market_features(
        bars, funding, oi, symbol=SYMBOL, timeframe=TIMEFRAME, oi_period="4h"
    )
    meta_features = build_v4aq_meta_features(bars, external_features)
    primary_signal, signal_diagnostics = v4ap.generate_donchian_breakout_signal(bars)
    train_end = int(len(bars) * TRAIN_FRACTION)
    forward_end = int(len(bars) * FORWARD_END_FRACTION)
    if train_end <= 100 or forward_end - train_end <= HORIZON_BARS + 2:
        raise ValueError("v4aq_research_split_invalid")
    training = _training_label_frame(bars, meta_features, primary_signal, train_end)
    model, model_diagnostics = _fit_meta_model(training, train_end)
    comparison, trades, decisions = _run_economic_comparison(
        bars.iloc[:forward_end].reset_index(drop=True),
        meta_features.iloc[:forward_end].reset_index(drop=True),
        primary_signal.iloc[:forward_end].reset_index(drop=True),
        model,
        model_diagnostics,
        train_end,
        forward_end,
    )
    report = _build_completed_report(
        commit_sha,
        spot_provenance,
        gap_diagnostics,
        coverage,
        model_diagnostics,
        comparison,
        training,
        external_diagnostics,
        signal_diagnostics,
    )
    return report, trades, decisions


def run_v4aq_ai6(
    input_dir: Path,
    external_dir: Path,
    output_report: Path,
    output_trades_csv: Path,
    output_decisions_csv: Path,
    *,
    commit_sha: str,
) -> dict[str, Any]:
    """Run coverage preflight and fit/evaluate only on a complete 5-year window."""
    _validate_run_request(commit_sha)
    manifests = {
        symbol: _read_symbol_manifest(external_dir, symbol, commit_sha)
        for symbol in ("BTCUSDT", "ETHUSDT")
    }
    coverage = _external_coverage(manifests)
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    if not _coverage_is_complete(coverage):
        report = _build_blocked_report(commit_sha, coverage)
    else:
        report, trades, decisions = _run_complete_experiment(
            input_dir, external_dir, commit_sha, coverage
        )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output_trades_csv,
        trades,
        ("candidate_id", "symbol", "timeframe", "strategy", "geometry", "net_pnl_usd"),
    )
    _write_csv(
        output_decisions_csv,
        decisions,
        (
            "event_index",
            "signal_timestamp",
            "meta_probability",
            "selected_threshold",
            "decision",
            "split_role",
            "executable_signal",
        ),
    )
    print(_render_summary(report), end="")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-trades-csv", type=Path, required=True)
    parser.add_argument("--output-decisions-csv", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    run_v4aq_ai6(
        args.input_dir,
        args.external_dir,
        args.output_report,
        args.output_trades_csv,
        args.output_decisions_csv,
        commit_sha=args.commit_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
