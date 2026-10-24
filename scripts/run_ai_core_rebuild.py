"""Run the isolated 4H AI-core research rebuild; never creates orders or approvals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    BASE_AI_FEATURE_COLUMNS,
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

OUTPUT_SCHEMA = "alix.ai-core-rebuild-report.v1"
FORWARD_HORIZON_4H = 6
ROUND_TRIP_COST = ROUND_TRIP_COST_BPS / 10_000.0
MINIMUM_TRAIN_ROWS = 360
FOLD_COUNT = 3
STRATEGY_IDS = ("trend_following", "mean_reversion", "momentum_breakout")


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
    if any(
        item.owner_approved or item.lifecycle.value != "RESEARCH_ONLY"
        for item in registry.strategies
    ):
        raise ValueError("ai_core_research_registry_must_remain_unapproved")
    return registry, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_eth_4h(
    input_path: Path, btc_4h: DataFrame, phase_minutes: int
) -> tuple[DataFrame, dict[str, Any]]:
    """Load a supplied and hash-pinned hourly CSV; this function never fetches data."""
    raw_bytes = input_path.read_bytes()
    frame = pd.read_csv(input_path)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError("ai_core_eth_input_columns_missing")
    parsed = pd.to_datetime(frame["date"], errors="raise")
    if parsed.dt.tz is None:
        raise ValueError("ai_core_eth_timestamps_must_be_timezone_aware")
    frame["date"] = parsed.dt.tz_convert("UTC")
    expected_first = pd.Timestamp(btc_4h["date"].iloc[0]) - pd.Timedelta(
        minutes=phase_minutes
    )
    expected_last = (
        pd.Timestamp(btc_4h["date"].iloc[-1])
        - pd.Timedelta(minutes=phase_minutes)
        + pd.Timedelta(hours=3)
    )
    frame = frame.loc[frame["date"].between(expected_first, expected_last)].copy()
    if len(frame) != 4 * len(btc_4h):
        raise ValueError("ai_core_eth_input_hourly_coverage_incomplete")
    hourly = frame.loc[:, ["date", "open", "high", "low", "close", "volume"]]
    eth_4h = aggregate_hourly_to_4h(hourly)
    eth_4h["date"] = eth_4h["date"] + pd.Timedelta(minutes=phase_minutes)
    if not eth_4h["date"].reset_index(drop=True).equals(
        btc_4h["date"].reset_index(drop=True)
    ):
        raise ValueError("ai_core_eth_cross_asset_labels_do_not_match_btc")
    return eth_4h, {
        "status": "LOADED_HASH_PINNED_INPUT",
        "provider": "owner-supplied historical CSV",
        "symbol": "ETHUSDT",
        "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "rows_hourly": len(hourly),
        "rows_4h": len(eth_4h),
        "timestamp_phase_label_shift_minutes": phase_minutes,
        "exact_same_window_claimed": False,
        "source_semantics": (
            "Hourly source windows are relabelled to the BTC source's stable timestamp "
            "phase for sensitivity only; matching labels do not prove identical venue windows."
        ),
    }


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


def _empty_strategy_summaries() -> dict[str, dict[str, Any]]:
    return {
        strategy_id: {"routed_rows": 0, "candidate_signals": 0, "net_returns": []}
        for strategy_id in STRATEGY_IDS
    }


def _evaluate_candidate_signals(
    test_rows: DataFrame,
    predictions: np.ndarray,
    regimes: tuple[str, ...],
    registry: Any,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    summaries = _empty_strategy_summaries()
    prediction_rows: list[dict[str, Any]] = []
    next_allowed = {name: 0 for name in summaries}
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
                    net_return = float(row["forward_return"] - ROUND_TRIP_COST)
                    raw_returns = summary["net_returns"]
                    if isinstance(raw_returns, list):
                        raw_returns.append(net_return)
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
                "realized_forward_return": float(row["forward_return"]),
                "research_trade_accepted": accepted,
                "route_or_signal_reason": reason,
                "Champion": None,
                "order_permission": False,
            }
        )
    for strategy_id, summary in summaries.items():
        values = summary.pop("net_returns")
        returns = values if isinstance(values, list) else []
        summary.update(_trade_stats(returns))
    return summaries, prediction_rows


def _evaluate_fold(
    data: DataFrame,
    regime_features: DataFrame,
    feature_names: tuple[str, ...],
    fold_index: int,
    train_stop: int,
    test_start: int,
    test_stop: int,
    registry: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = data.iloc[:train_stop]
    test = data.iloc[test_start:test_stop]
    train_features = train.loc[:, list(feature_names)]
    test_features = test.loc[:, list(feature_names)]
    ensemble = fit_weighted_ensemble(
        train_features,
        train["forward_return"],
        test_features,
        config=EnsembleConfig(n_estimators=48, random_seed=42 + fold_index),
    )
    actual = test["forward_return"].to_numpy(dtype=float)
    predictions = np.asarray(ensemble.predictions, dtype=float)
    train_regimes = regime_features.loc[train.index]
    if len(train_regimes) < 100:
        raise ValueError("ai_core_hmm_training_rows_below_minimum")
    fitted_hmm = fit_gaussian_hmm(
        train_regimes, max_iter=24, random_seed=42 + fold_index
    )
    causal_window = regime_features.loc[train.index[0] : test.index[-1]]
    causal_regimes, posterior = infer_hmm_regimes(causal_window, fitted_hmm)
    if len(posterior) != len(causal_window):
        raise ValueError("ai_core_hmm_posterior_accounting_mismatch")
    regime_by_index = dict(zip(causal_window.index, causal_regimes, strict=True))
    test_regime_names = tuple(regime_by_index[index] for index in test.index)
    strategy_summary, rows = _evaluate_candidate_signals(
        test, predictions, test_regime_names, registry
    )
    component_metrics = {
        name: regression_metrics(
            actual, np.asarray(component_predictions, dtype=float)
        )
        for name, component_predictions in ensemble.per_model_predictions
    }
    report = {
        "fold": fold_index,
        "training_rows": len(train),
        "training_stop_exclusive": train_stop,
        "test_start": test_start,
        "test_stop_exclusive": test_stop,
        "purge_horizon_bars": FORWARD_HORIZON_4H,
        "training_end_precedes_test_by_horizon": (
            train_stop <= test_start - FORWARD_HORIZON_4H
        ),
        "ensemble_metrics": regression_metrics(actual, predictions),
        "component_model_metrics": component_metrics,
        "hmm_mean_log_likelihood": fitted_hmm.mean_log_likelihood,
        "hmm_iterations": fitted_hmm.iterations,
        "hmm_state_occupancy_training": dict(
            zip(
                ("trend", "range", "high_volatility"),
                fitted_hmm.state_occupancy,
                strict=True,
            )
        ),
        "strategy_research_diagnostics": strategy_summary,
    }
    return report, rows


def _summarize_strategies(
    fold_reports: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for strategy_id in STRATEGY_IDS:
        fold_stats = [
            report["strategy_research_diagnostics"][strategy_id]
            for report in fold_reports
        ]
        trade_count = sum(int(item["trade_count"]) for item in fold_stats)
        expectancy_sum = sum(
            float(item["net_expectancy"]) * int(item["trade_count"])
            for item in fold_stats
            if item["net_expectancy"] is not None
        )
        result[strategy_id] = {
            "routed_rows": sum(int(item["routed_rows"]) for item in fold_stats),
            "candidate_signals": sum(int(item["candidate_signals"]) for item in fold_stats),
            "trade_count": trade_count,
            "net_expectancy": expectancy_sum / trade_count if trade_count else None,
            "max_abs_drawdown_max_across_folds": max(
                float(item["max_abs_drawdown"]) for item in fold_stats
            ),
        }
    return result


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


def _prepare_market_dataset(
    input_dir: Path, eth_input: Path | None
) -> tuple[DataFrame, DataFrame, tuple[str, ...], dict[str, Any], dict[str, Any], str]:
    hourly_btc, source_report, segment_hash = stage_b._load_development(input_dir)
    phases = tuple(
        sorted(pd.to_datetime(hourly_btc["date"], utc=True).dt.minute.unique().tolist())
    )
    if len(phases) != 1:
        raise ValueError("ai_core_btc_timestamp_phase_not_uniform")
    phase_minutes = int(phases[0])
    btc_4h = aggregate_hourly_to_4h(hourly_btc)
    if eth_input is None:
        features = build_ai_features(btc_4h)
        feature_names = BASE_AI_FEATURE_COLUMNS
        eth_manifest: dict[str, Any] = {
            "status": "BLOCKED_MISSING_APPROVED_CROSS_ASSET_DATASET",
            "provider_fetch_performed": False,
            "reason": (
                "Cross-asset features are implemented but the repository currently contains "
                "no pinned, approved ETH hourly dataset. Supply --eth-input to evaluate them."
            ),
        }
    else:
        eth_4h, eth_manifest = _load_eth_4h(eth_input, btc_4h, phase_minutes)
        features = build_ai_features(btc_4h, cross_asset={"ETHUSDT": eth_4h})
        feature_names = AI_FEATURE_COLUMNS
    forward_return = btc_4h["close"].shift(-FORWARD_HORIZON_4H) / btc_4h["close"] - 1.0
    valid = (
        features.loc[:, list(feature_names)].notna().all(axis=1)
        & forward_return.replace([np.inf, -np.inf], np.nan).notna()
    )
    data = features.loc[valid].copy()
    data["forward_return"] = forward_return.loc[valid]
    regime_features = calculate_regime_features(btc_4h).loc[data.index]
    source = {
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
    }
    return data, regime_features, feature_names, source, eth_manifest, segment_hash


def _run_walk_forward(
    data: DataFrame,
    regime_features: DataFrame,
    feature_names: tuple[str, ...],
    registry: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_specs = _walk_forward_folds(
        len(data),
        min_train_rows=MINIMUM_TRAIN_ROWS,
        horizon=FORWARD_HORIZON_4H,
        folds=FOLD_COUNT,
    )
    reports: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for fold_index, (train_stop, test_start, test_stop) in enumerate(fold_specs, start=1):
        fold_report, rows = _evaluate_fold(
            data,
            regime_features,
            feature_names,
            fold_index,
            train_stop,
            test_start,
            test_stop,
            registry,
        )
        reports.append(fold_report)
        prediction_rows.extend(rows)
    return reports, prediction_rows


def _report_document(
    commit_sha: str,
    source: dict[str, Any],
    feature_names: tuple[str, ...],
    registry: Any,
    registry_sha: str,
    fold_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    actual_cross_asset = source["eth_cross_asset_manifest"]["status"] == "LOADED_HASH_PINNED_INPUT"
    safety = {
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
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "run_status": "COMPLETED_RESEARCH_ONLY",
        "created_by_commit_sha": commit_sha,
        "source": source,
        "feature_engineering": {
            "schema_version": AI_FEATURE_SCHEMA_VERSION,
            "model_feature_count": len(feature_names),
            "model_feature_names": list(feature_names),
            "extended_feature_schema_count": len(AI_FEATURE_COLUMNS),
            "groups": [
                "returns",
                "SMA_and_EMA",
                "Bollinger_bands",
                "multi_window_RSI",
                "ATR_and_true_range",
                "realized_and_downside_volatility",
                "volume_zscores_and_turnover",
                "OHLCV_volume_profile_proxy",
                "ETH_cross_asset_return_correlation_and_beta",
            ],
            "volume_profile_limitation": (
                "Profile features are coarse bins of candle typical price weighted by OHLCV "
                "volume, not trade-level volume-at-price data."
            ),
            "cross_asset_status": (
                "INCLUDED_WITH_TIMESTAMP_PHASE_CAVEAT"
                if actual_cross_asset
                else "BLOCKED_MISSING_APPROVED_CROSS_ASSET_DATASET"
            ),
            "cross_asset_limitation": (
                "ETHUSDT labels are phase-aligned for sensitivity only; matching timestamps "
                "do not establish exact same-window or same-venue observations."
            ),
            "lookahead_policy": (
                "trailing windows only; exact timestamp join for optional cross-asset input; "
                "no backfill or interpolation"
            ),
        },
        "ensemble": {
            "models": ["xgboost", "random_forest", "lightgbm"],
            "method": "equal_weighted_mean",
            "weights": {
                "xgboost": 1.0 / 3.0,
                "random_forest": 1.0 / 3.0,
                "lightgbm": 1.0 / 3.0,
            },
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
                "high_volatility": "momentum_breakout",
            },
            "production_registry_modified": False,
            "order_permission": False,
        },
        "strategy_research_diagnostics": _summarize_strategies(fold_reports),
        "sentiment": {
            "module_available": True,
            "provider_fetch_performed": False,
            "used_in_model_features": False,
            "status": "BLOCKED_PENDING_APPROVED_PERSIAN_RSS_PROVIDER_AND_CALIBRATED_MODEL",
            "reason": (
                "Point-in-time RSS parsing and aggregation are implemented; no provider "
                "license/retention approval or calibrated owner-reviewed NLP model is approved."
            ),
        },
        "economic_evaluation": {
            "status": "NOT_EVALUATED_RESEARCH_ONLY",
            "economic_acceptance": "NO-GO",
            "reason": (
                "Technical metrics are diagnostics only and are not owner-approved economic evidence."
            ),
        },
        "safety_invariants": safety,
    }


def run_rebuild(
    input_dir: Path,
    output_dir: Path,
    commit_sha: str,
    *,
    eth_input: Path | None = None,
) -> dict[str, Any]:
    if len(commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in commit_sha
    ):
        raise ValueError("ai_core_commit_sha_invalid")
    project_root = Path(__file__).resolve().parents[1]
    registry, registry_sha = _load_research_registry(project_root)
    data, regime_features, feature_names, source, _, segment_hash = _prepare_market_dataset(
        input_dir, eth_input
    )
    if len(data) <= MINIMUM_TRAIN_ROWS + FORWARD_HORIZON_4H + FOLD_COUNT:
        raise ValueError("ai_core_research_rows_insufficient")
    source["segment_a_rows_sha256"] = segment_hash
    fold_reports, predictions = _run_walk_forward(
        data, regime_features, feature_names, registry
    )
    report = _report_document(
        commit_sha, source, feature_names, registry, registry_sha, fold_reports
    )
    safety = report["safety_invariants"]
    if safety["Champion"] is not None or safety["order_permission"] is not False:
        raise ValueError("ai_core_safety_invariant_violated")
    _write_json(output_dir / "ai-core-rebuild-report.json", report)
    _write_csv(output_dir / "ai-core-oos-predictions.csv", predictions)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument(
        "--eth-input",
        type=Path,
        default=None,
        help="Optional owner-supplied, approved, hash-pinned hourly ETHUSDT CSV; no download is performed.",
    )
    args = parser.parse_args()
    report = run_rebuild(
        args.input_dir,
        args.output_dir,
        args.commit_sha,
        eth_input=args.eth_input,
    )
    print(
        json.dumps(
            {
                "commit_sha": report["created_by_commit_sha"],
                "run_status": report["run_status"],
                "timeframe": report["source"]["evaluation_timeframe"],
                "model_feature_count": report["feature_engineering"]["model_feature_count"],
                "ensemble_models": report["ensemble"]["models"],
                "strategies": report["strategy_registry"]["candidate_count"],
                "oos_predictions": sum(
                    item["test_stop_exclusive"] - item["test_start"]
                    for item in report["ensemble"]["walk_forward_folds"]
                ),
                "cross_asset_status": report["feature_engineering"]["cross_asset_status"],
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
