"""AI-6 Funding-only Meta-Labeling 36-cell economic evaluation."""

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

from scripts import run_ml_core_v4ap_diagnostic_sweep as v4ap
from services.meta_labeling import v4x
from services.ml_core.economic_backtest import resample_ohlcv
from services.meta_labeling.v4aq import (
    V4AQ_META_FEATURE_COLUMNS,
    build_v4aq_meta_features,
    fit_v4aq_lightgbm_meta_filter,
    predict_v4aq_meta_probabilities,
)
from services.ml_core.external_market_features import (
    EXTERNAL_FEATURE_COLUMNS,
    build_external_market_features,
)

REPORT_SCHEMA_VERSION = "alix.ml-core.v4aq-ai6.v2"
REQUESTED_DAYS = 1825
TRAIN_FRACTION = 0.70
FORWARD_END_FRACTION = 0.85
OOF_FOLD_COUNT = 5
TARGET_ACCEPT_FRACTION = 0.50
ROUND_TRIP_COST_BPS = 30.0
FEE_BPS_PER_SIDE = 10.0
SLIPPAGE_BPS_PER_SIDE = 5.0
ROC_AUC_REFERENCE = 0.51
MINIMUM_TRADE_COUNT = 50
MINIMUM_WIN_RATE = 0.40
MINIMUM_NET_PROFIT_FACTOR = 1.10
MINIMUM_DSR_Z = 0.0
FUNDING_CSV_COLUMNS = (
    "symbol",
    "funding_time",
    "funding_rate",
    "mark_price",
    "available_at",
    "content_hash",
)
SAFETY = {
    "Champion": None,
    "order_permission": False,
    "trading_authorization": "NO-GO",
    "private_api_calls": 0,
    "real_order_attempts": 0,
}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate_commit_sha(commit_sha: str) -> None:
    if len(commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in commit_sha
    ):
        raise ValueError("ai6_commit_sha_invalid")


def _validate_funding_manifest_contract(
    raw: dict[str, Any], symbol: str, commit_sha: str
) -> dict[str, Any]:
    if "open_interest" in raw or "open_interest_csv" in raw:
        raise ValueError(f"ai6_oi_contract_must_be_absent:{symbol}")
    expected = (
        raw.get("schema_version") == "alix.binance.usdm-funding.v1",
        raw.get("commit_sha") == commit_sha,
        raw.get("provider") == "Binance USD-M public Funding Rate History",
        raw.get("host") == "fapi.binance.com",
        raw.get("endpoint") == "/fapi/v1/fundingRate",
        raw.get("symbol") == symbol,
        raw.get("requested_days") == REQUESTED_DAYS,
        raw.get("public_api_only") is True,
        raw.get("credentials_used") is False,
        raw.get("api_key_header_sent") is False,
        raw.get("private_api_calls") == 0,
        raw.get("exchange_order_calls") == 0,
        raw.get("Champion") is None,
        raw.get("order_permission") is False,
        raw.get("trading_authorization") == "NO-GO",
        raw.get("requested_start_utc") is not None,
        raw.get("requested_end_utc") is not None,
    )
    if not all(expected):
        raise ValueError(f"ai6_funding_manifest_contract_invalid:{symbol}")
    funding_meta = raw.get("funding")
    if not isinstance(funding_meta, dict):
        raise ValueError(f"ai6_funding_manifest_section_invalid:{symbol}")
    return funding_meta


def _validate_funding_manifest_window(
    raw: dict[str, Any],
    symbol: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> None:
    requested_start = pd.Timestamp(raw["requested_start_utc"])
    requested_end = pd.Timestamp(raw["requested_end_utc"])
    matches = (
        requested_start.tzinfo is not None
        and requested_end.tzinfo is not None
        and requested_start.tz_convert("UTC") == start_utc
        and requested_end.tz_convert("UTC") == end_utc
    )
    if not matches:
        raise ValueError(f"ai6_funding_manifest_window_mismatch:{symbol}")


def _load_funding_manifest(
    external_dir: Path,
    symbol: str,
    commit_sha: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    stem = f"binance-{symbol.lower()}"
    manifest_path = external_dir / f"{stem}-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"ai6_funding_manifest_missing:{symbol}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"ai6_funding_manifest_schema_invalid:{symbol}")
    funding_meta = _validate_funding_manifest_contract(raw, symbol, commit_sha)
    _validate_funding_manifest_window(raw, symbol, start_utc, end_utc)
    csv_name = raw.get("funding_csv")
    if csv_name != f"{stem}-funding.csv":
        raise ValueError(f"ai6_funding_csv_name_invalid:{symbol}")
    return raw, funding_meta, str(csv_name)


def _read_funding_csv(
    external_dir: Path,
    symbol: str,
    csv_name: str,
    raw: dict[str, Any],
    funding_meta: dict[str, Any],
) -> DataFrame:
    csv_path = external_dir / csv_name
    if not csv_path.is_file():
        raise ValueError(f"ai6_funding_csv_missing:{symbol}")
    if _sha256(csv_path) != raw.get("funding_csv_sha256"):
        raise ValueError(f"ai6_funding_csv_hash_mismatch:{symbol}")
    funding = pd.read_csv(csv_path)
    if tuple(funding.columns) != FUNDING_CSV_COLUMNS:
        raise ValueError(f"ai6_funding_csv_schema_invalid:{symbol}")
    if len(funding) != funding_meta.get("row_count"):
        raise ValueError(f"ai6_funding_row_count_mismatch:{symbol}")
    return funding


def _validate_funding_csv_values(funding: DataFrame, symbol: str) -> None:
    if not funding["symbol"].eq(symbol).all():
        raise ValueError(f"ai6_funding_symbol_mismatch:{symbol}")
    funding["funding_time"] = pd.to_datetime(
        funding["funding_time"], format="mixed", utc=True, errors="coerce"
    )
    funding["available_at"] = pd.to_datetime(
        funding["available_at"], format="mixed", utc=True, errors="coerce"
    )
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="coerce")
    invalid = (
        funding[["funding_time", "available_at", "funding_rate"]].isna().any().any()
        or not np.isfinite(funding["funding_rate"].to_numpy(dtype=float)).all()
        or funding["funding_time"].duplicated().any()
        or not funding["funding_time"].is_monotonic_increasing
    )
    if invalid:
        raise ValueError(f"ai6_funding_csv_values_invalid:{symbol}")


def _funding_coverage_ready(raw: dict[str, Any], funding_meta: dict[str, Any]) -> bool:
    return bool(
        raw.get("research_status") == "READY_FOR_FUNDING_FEATURE_JOIN"
        and funding_meta.get("full_requested_history_available") is True
        and funding_meta.get("coverage_status") == "FULL_REQUESTED_HISTORY"
        and int(funding_meta.get("row_count", 0)) >= 20
    )


def _read_funding_artifact(
    external_dir: Path,
    symbol: str,
    commit_sha: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[DataFrame, dict[str, Any], bool]:
    """Verify one Funding CSV and its source manifest; reject all OI-era schemas."""
    raw, funding_meta, csv_name = _load_funding_manifest(
        external_dir, symbol, commit_sha, start_utc, end_utc
    )
    funding = _read_funding_csv(external_dir, symbol, csv_name, raw, funding_meta)
    if funding.empty:
        return funding, raw, False
    _validate_funding_csv_values(funding, symbol)
    return funding, raw, _funding_coverage_ready(raw, funding_meta)


def _label_primary_events(
    bars: DataFrame,
    events: DataFrame,
    *,
    timeframe: str,
    horizon_bars: int,
) -> DataFrame:
    """Label OOF BUY events by fixed 24h net close return using the 30bp cost model."""
    if events.empty or "event_index" not in events.columns:
        return DataFrame()
    if timeframe not in v4ap.TIMEFRAME_MINUTES or horizon_bars <= 0:
        raise ValueError("ai6_label_horizon_configuration_invalid")
    indices = events["event_index"].to_numpy(dtype=int)
    end_indices = indices + horizon_bars
    row_count = len(bars)
    valid = (indices >= 0) & (end_indices < row_count)
    dates = pd.to_datetime(bars["date"], utc=True).astype("int64").to_numpy()
    interval_ns = int(pd.Timedelta(minutes=v4ap.TIMEFRAME_MINUTES[timeframe]).value)
    gap_flags = np.zeros(row_count, dtype=np.int64)
    if row_count > 1:
        gap_flags[1:] = np.diff(dates) != interval_ns
    gap_prefix = np.cumsum(gap_flags)
    safe_positions = np.flatnonzero(valid)
    if safe_positions.size:
        valid[safe_positions] &= (
            dates[end_indices[safe_positions]] - dates[indices[safe_positions]]
            == horizon_bars * interval_ns
        )
        valid[safe_positions] &= (
            gap_prefix[end_indices[safe_positions]]
            - gap_prefix[indices[safe_positions]]
            == 0
        )
    safe_positions = np.flatnonzero(valid)
    if safe_positions.size == 0:
        return DataFrame()
    event_indices = indices[safe_positions]
    final_indices = end_indices[safe_positions]
    opens = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    entry_reference = opens[event_indices + 1]
    exit_reference = closes[final_indices]
    price_valid = (
        np.isfinite(entry_reference)
        & np.isfinite(exit_reference)
        & (entry_reference > 0.0)
        & (exit_reference > 0.0)
    )
    if not price_valid.all():
        safe_positions = safe_positions[price_valid]
        event_indices = event_indices[price_valid]
        final_indices = final_indices[price_valid]
        entry_reference = entry_reference[price_valid]
        exit_reference = exit_reference[price_valid]
    if safe_positions.size == 0:
        return DataFrame()
    fee_rate = FEE_BPS_PER_SIDE / 10_000.0
    slippage_rate = SLIPPAGE_BPS_PER_SIDE / 10_000.0
    entry_fill = entry_reference * (1.0 + slippage_rate)
    exit_fill = exit_reference * (1.0 - slippage_rate)
    entry_fee = entry_fill * fee_rate
    exit_fee = exit_fill * fee_rate
    net_return = (exit_fill - entry_fill - entry_fee - exit_fee) / entry_reference
    result = events.iloc[safe_positions].copy().reset_index(drop=True)
    result["label_end_idx"] = final_indices.astype(int)
    result["net_return_fraction"] = net_return.astype(float)
    result["meta_label"] = (net_return > 0.0).astype(int)
    return result


def _evaluate_primary_group(
    bars: DataFrame,
    features: DataFrame,
    primary_signal: Series,
    *,
    strategy: str,
    timeframe: str,
    train_end: int,
    forward_end: int,
) -> tuple[Series, dict[str, Any], list[dict[str, Any]]]:
    """Fit one purged OOF model and apply its frozen threshold to forward BUY events."""
    horizon_bars = v4ap.HORIZON_HOURS * 60 // v4ap.TIMEFRAME_MINUTES[timeframe]
    empty = Series(np.zeros(len(bars), dtype=bool), index=bars.index, dtype=bool)
    try:
        stop_index = train_end - horizon_bars
        if stop_index <= 5:
            raise ValueError("ai6_training_window_too_short")
        oof_events = v4x.generate_oof_primary_events(
            primary_signal.reset_index(drop=True),
            start_index=0,
            stop_index=stop_index,
            fold_count=OOF_FOLD_COUNT,
        )
        labelled = _label_primary_events(
            bars,
            oof_events,
            timeframe=timeframe,
            horizon_bars=horizon_bars,
        )
        if labelled.empty:
            raise ValueError("ai6_oof_training_labels_empty")
        feature_values = features.iloc[
            labelled["event_index"].to_numpy(dtype=int)
        ].to_numpy(dtype=float)
        complete = np.isfinite(feature_values).all(axis=1)
        labelled = labelled.loc[complete].reset_index(drop=True)
        if labelled.empty:
            raise ValueError("ai6_oof_funding_features_incomplete")
        fit_events, calibration_events = v4x.split_meta_fit_calibration(
            labelled,
            outer_train_end_index=train_end,
        )
        fit_features = features.iloc[
            fit_events["event_index"].to_numpy(dtype=int)
        ].reset_index(drop=True)
        calibration_features = features.iloc[
            calibration_events["event_index"].to_numpy(dtype=int)
        ].reset_index(drop=True)
        model, fit_diagnostics = fit_v4aq_lightgbm_meta_filter(
            fit_features,
            Series(fit_events["meta_label"].to_numpy(dtype=int)),
            calibration_features,
            target_accept_fraction=TARGET_ACCEPT_FRACTION,
        )
    except ValueError as error:
        status = f"BLOCKED_TRAINING_SUPPORT:{error}"
        return (
            empty,
            {
                "fit_performed": False,
                "fit_status": status,
                "fit_rows": 0,
                "calibration_rows": 0,
                "oof_event_count": 0,
                "validation_event_count": 0,
                "validation_labeled_event_count": 0,
                "roc_auc": None,
                "roc_auc_status": "NOT_EVALUATED_ROC_AUC_DEFERRED_FOR_ECONOMIC_SWEEP",
                "threshold": None,
                "accepted_event_count": 0,
                "acceptance_rate": 0.0,
            },
            [],
        )

    validation_stop = max(train_end, forward_end - horizon_bars)
    validation_indices = (
        np.flatnonzero(
            primary_signal.iloc[train_end:validation_stop].to_numpy(dtype=bool)
        )
        + train_end
    )
    validation_events = DataFrame(
        {
            "event_index": validation_indices.astype(int),
            "primary_signal": "BUY",
            "primary_oof": False,
            "fold_id": -1,
        }
    )
    labelled_validation = _label_primary_events(
        bars,
        validation_events,
        timeframe=timeframe,
        horizon_bars=horizon_bars,
    )
    if labelled_validation.empty:
        probabilities = np.asarray([], dtype=float)
        labels = np.asarray([], dtype=int)
        validation_features = DataFrame(columns=V4AQ_META_FEATURE_COLUMNS)
    else:
        feature_values = features.iloc[
            labelled_validation["event_index"].to_numpy(dtype=int)
        ].to_numpy(dtype=float)
        complete = np.isfinite(feature_values).all(axis=1)
        labelled_validation = labelled_validation.loc[complete].reset_index(drop=True)
        validation_features = features.iloc[
            labelled_validation["event_index"].to_numpy(dtype=int)
        ].reset_index(drop=True)
        probabilities = (
            predict_v4aq_meta_probabilities(model, validation_features)
            if len(validation_features)
            else np.asarray([], dtype=float)
        )
        labels = labelled_validation["meta_label"].to_numpy(dtype=int)
    auc = None
    auc_status = "NOT_EVALUATED_ROC_AUC_DEFERRED_FOR_ECONOMIC_SWEEP"
    threshold = float(fit_diagnostics["selected_threshold"])
    accepted = probabilities >= threshold
    accepted_indices = labelled_validation["event_index"].to_numpy(dtype=int)[accepted]
    filtered = np.zeros(len(bars), dtype=bool)
    if accepted_indices.size:
        filtered[accepted_indices] = True
    decision_rows = [
        {
            "event_index": int(event_index),
            "probability": float(probability),
            "selected_threshold": threshold,
            "meta_label": int(label),
            "net_target_return_fraction": float(net_return),
            "accepted": bool(selected),
            "target_horizon_hours": v4ap.HORIZON_HOURS,
        }
        for event_index, probability, label, net_return, selected in zip(
            labelled_validation["event_index"].to_numpy(dtype=int),
            probabilities,
            labels,
            labelled_validation["net_return_fraction"].to_numpy(dtype=float),
            accepted,
        )
    ]
    result = {
        "fit_performed": True,
        "fit_status": "FITTED",
        "fit_rows": int(fit_diagnostics["fit_rows"]),
        "calibration_rows": int(fit_diagnostics["calibration_rows"]),
        "oof_event_count": int(len(labelled)),
        "validation_event_count": int(len(validation_indices)),
        "validation_labeled_event_count": int(len(labelled_validation)),
        "roc_auc": auc,
        "roc_auc_status": auc_status,
        "threshold": threshold,
        "accepted_event_count": int(accepted.sum()),
        "acceptance_rate": (
            float(accepted.sum() / len(accepted)) if len(accepted) else 0.0
        ),
        "fit_diagnostics": fit_diagnostics,
    }
    return Series(filtered, index=bars.index, dtype=bool), result, decision_rows


def _placeholder_candidate(
    symbol: str,
    timeframe: str,
    strategy: str,
    geometry: dict[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    candidate = {
        "candidate_id": v4ap._candidate_id(
            symbol, timeframe, strategy, str(geometry["name"])
        ),
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy,
        "geometry": str(geometry["name"]),
        "bars_evaluated": 0,
        "raw_signal_count": 0,
        "eligible_entry_signal_count": 0,
        "gap_flatten_count": 0,
        "trade_count": 0,
        "win_rate": 0.0,
        "net_profit_factor": 0.0,
        "net_profit_factor_infinite_no_losses": False,
        "max_drawdown_percent": 0.0,
        "net_pnl_usd": 0.0,
        "net_expectancy_usd_per_trade": None,
        "total_fees_usd": 0.0,
        "total_slippage_cost_usd": 0.0,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "fee_bps_per_side": FEE_BPS_PER_SIDE,
        "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
        "stop_loss_atr": float(geometry["stop_loss_atr"]),
        "take_profit_atr": float(geometry["take_profit_atr"]),
        "trailing_enabled": bool(geometry["trailing_enabled"]),
        "trailing_activation_atr": geometry["trailing_activation_atr"],
        "trailing_step_atr": geometry["trailing_step_atr"],
        "horizon_hours_nominal": v4ap.HORIZON_HOURS,
        "horizon_bars": v4ap.HORIZON_HOURS * 60 // v4ap.TIMEFRAME_MINUTES[timeframe],
        "same_candle_collision_policy": "STOP_LOSS_WINS",
        "execution_policy": "CLOSED_BAR_NEXT_BAR_OPEN",
        "return_observation_count": 0,
        "sharpe_ratio_per_trade": None,
        "return_skewness": None,
        "return_kurtosis_non_excess": None,
        "sharpe_status": "NO_TRADES_DSR_UNDEFINED",
        "roc_auc": None,
        "roc_auc_reference": ROC_AUC_REFERENCE,
        "roc_auc_status": "NOT_EVALUATED_ROC_AUC_DEFERRED_FOR_ECONOMIC_SWEEP",
        "roc_auc_above_reference": None,
        "unfiltered_primary_signal_count": 0,
        "meta_filtered_signal_count": 0,
        "meta_filter_acceptance_rate": 0.0,
        "meta_model_fit_status": status,
        "meta_model_fit_rows": 0,
        "meta_model_calibration_rows": 0,
        "validation_event_count": 0,
        "validation_labeled_event_count": 0,
        "meta_threshold": None,
        "qualifies": False,
    }
    return candidate


def _candidate_acceptance(candidate: dict[str, Any]) -> dict[str, bool]:
    trade_count_passed = int(candidate.get("trade_count", 0)) >= MINIMUM_TRADE_COUNT
    win_rate = candidate.get("win_rate")
    win_rate_passed = win_rate is not None and float(win_rate) >= MINIMUM_WIN_RATE
    profit_factor = candidate.get("net_profit_factor")
    pf_passed = candidate.get("net_profit_factor_infinite_no_losses") is True or (
        profit_factor is not None and float(profit_factor) >= MINIMUM_NET_PROFIT_FACTOR
    )
    dsr = candidate.get("dsr_z_score")
    dsr_passed = dsr is not None and float(dsr) > MINIMUM_DSR_Z
    return {
        "trade_count_passed": bool(trade_count_passed),
        "win_rate_passed": bool(win_rate_passed),
        "net_profit_factor_passed": bool(pf_passed),
        "dsr_passed": bool(dsr_passed),
        "statistical_edge_accepted": bool(
            trade_count_passed and win_rate_passed and pf_passed and dsr_passed
        ),
    }


def _candidate_has_net_pf_gate(candidate: dict[str, Any]) -> bool:
    if candidate.get("net_profit_factor_infinite_no_losses") is True:
        return True
    profit_factor = candidate.get("net_profit_factor")
    return (
        profit_factor is not None and float(profit_factor) >= MINIMUM_NET_PROFIT_FACTOR
    )


def _report_decision(
    candidates: list[dict[str, Any]],
    model_groups: list[dict[str, Any]],
    status: str,
) -> tuple[str, list[str], list[str], int]:
    qualifying = [str(item["candidate_id"]) for item in candidates if item["qualifies"]]
    pf_candidates = [
        str(item["candidate_id"])
        for item in candidates
        if _candidate_has_net_pf_gate(item)
    ]
    model_fit_count = sum(bool(item.get("fit_performed")) for item in model_groups)
    if status == "BLOCKED_INSUFFICIENT_FUNDING_HISTORY":
        conclusion = "FUNDING_COVERAGE_INSUFFICIENT_NO_ECONOMIC_CONCLUSION"
    elif qualifying:
        conclusion = "FUNDING_EDGE_ACCEPTED_UNDER_ALL_PREREGISTERED_GATES"
    elif pf_candidates:
        conclusion = "NET_PF_GATE_MET_BUT_FULL_STATISTICAL_ACCEPTANCE_FAILED"
    else:
        conclusion = "NO_NET_PF_EDGE_FOUND_IN_FUNDING_ONLY_FEATURES"
    return conclusion, qualifying, pf_candidates, model_fit_count


def _report_dataset_block(
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    market_provenance: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "requested_days": REQUESTED_DAYS,
        "source_interval": "1m",
        "start_utc": v4ap._iso_utc(start_utc),
        "end_utc": v4ap._iso_utc(end_utc),
        "assets": market_provenance,
        "analysis_timeframes": list(v4ap.TIMEFRAMES),
        "train_fraction": TRAIN_FRACTION,
        "forward_validation_end_fraction": FORWARD_END_FRACTION,
        "final_holdout_fraction": 1.0 - FORWARD_END_FRACTION,
        "final_holdout_status": "UNTOUCHED_NOT_USED_FOR_FIT_THRESHOLD_OR_ECONOMIC_DECISION",
        "missing_candles": "NEVER_INTERPOLATED_OR_SYNTHETICALLY_FILLED",
    }


def _report_funding_block(
    funding_provenance: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "provider": "Binance USD-M public Funding Rate History",
        "host": "fapi.binance.com",
        "endpoint": "/fapi/v1/fundingRate",
        "feature_columns": list(EXTERNAL_FEATURE_COLUMNS),
        "available_at_policy": "fundingTime + 5 minutes (conservative research proxy)",
        "maximum_observation_age": "2 days",
        "coverage": funding_provenance,
        "open_interest_used": False,
    }


def _report_model_block(
    model_groups: list[dict[str, Any]], model_fit_count: int
) -> dict[str, Any]:
    return {
        "family": "LightGBM binary classifier via existing V4-X purged OOF/calibration contract",
        "fit_performed": model_fit_count > 0,
        "fit_group_count": model_fit_count,
        "expected_fit_group_count": (
            len(v4ap.ASSETS) * len(v4ap.TIMEFRAMES) * len(v4ap.STRATEGIES)
        ),
        "groups": model_groups,
        "features": list(V4AQ_META_FEATURE_COLUMNS),
        "external_features": list(EXTERNAL_FEATURE_COLUMNS),
        "training_target": (
            "PRIMARY_BUY_ONLY_FIXED_24H_FORWARD_CLOSE_NET_RETURN_AFTER_30BPS_COSTS"
        ),
        "validation_metric": "NOT_EVALUATED_ROC_AUC_DEFERRED_FOR_ECONOMIC_SWEEP",
        "threshold_policy": (
            "SELECTED_FROM_PURGED_IN_TRAIN_CALIBRATION_PROBABILITIES_ONLY; "
            "NO_FORWARD_VALIDATION_LABELS_USED_FOR_THRESHOLD"
        ),
        "scope": (
            "ONE_MODEL_PER_ASSET_TIMEFRAME_PRIMARY_STRATEGY; "
            "SAME_FORWARD_PREDICTIONS_REUSED_FOR_THREE_EXIT_GEOMETRIES"
        ),
    }


def _report_economic_results(
    status: str, candidates: list[dict[str, Any]], model_fit_count: int
) -> dict[str, Any] | None:
    if status == "BLOCKED_INSUFFICIENT_FUNDING_HISTORY":
        return None
    return {
        "candidate_count": len(candidates),
        "total_trade_count": sum(int(item["trade_count"]) for item in candidates),
        "meta_models_fitted": model_fit_count,
        "dsr_trial_count": v4ap.DSR_TRIAL_COUNT,
        "economic_scope": "FORWARD_VALIDATION_ONLY; FINAL_15_PERCENT_UNTOUCHED",
    }


def _build_report(
    *,
    commit_sha: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    market_provenance: dict[str, dict[str, Any]],
    funding_provenance: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
    dsr_cohort: dict[str, Any],
    model_groups: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    conclusion, qualifying, pf_candidates, model_fit_count = _report_decision(
        candidates, model_groups, status
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "commit_sha": commit_sha,
        "status": status,
        "diagnostic": "V4-AQ_FUNDING_ONLY_META_FILTERED_36_CELL_SWEEP",
        "dataset": _report_dataset_block(start_utc, end_utc, market_provenance),
        "funding": _report_funding_block(funding_provenance),
        "preregistered_grid": {
            "assets": list(v4ap.ASSETS),
            "timeframes": list(v4ap.TIMEFRAMES),
            "strategies": list(v4ap.STRATEGIES),
            "exit_geometries": [dict(item) for item in v4ap.EXIT_GEOMETRIES],
            "expected_combinations": v4ap.DSR_TRIAL_COUNT,
        },
        "model": _report_model_block(model_groups, model_fit_count),
        "execution_assumptions": {
            "market": "BINANCE_SPOT_PUBLIC_OHLCV",
            "direction": "SPOT_LONG_ONLY",
            "signal_timing": "COMPLETED_BAR_CLOSE_NEXT_BAR_OPEN",
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "horizon_hours_nominal": v4ap.HORIZON_HOURS,
            "same_candle_collision_policy": "STOP_LOSS_WINS",
            "position_fraction": 1.0,
            "pyramiding": False,
        },
        "decision_rule": {
            "minimum_trade_count": MINIMUM_TRADE_COUNT,
            "minimum_win_rate": MINIMUM_WIN_RATE,
            "minimum_net_profit_factor": MINIMUM_NET_PROFIT_FACTOR,
            "minimum_dsr_z_exclusive": MINIMUM_DSR_Z,
            "roc_auc_reference": ROC_AUC_REFERENCE,
            "profit_factor_is_not_sufficient_without_trade_count_win_rate_and_dsr": True,
        },
        "dsr_cohort": dsr_cohort,
        "candidates": candidates,
        "qualifying_candidate_ids": qualifying,
        "net_pf_threshold_candidate_ids": pf_candidates,
        "conclusion": conclusion,
        "economic_results": _report_economic_results(
            status, candidates, model_fit_count
        ),
        "model_usage": {
            "meta_model_used": True,
            "funding_features_used": True,
            "open_interest_used": False,
            "hmm_used": False,
            "cusum_used": False,
            "lstm_used": False,
            "reinforcement_learning_used": False,
        },
        **SAFETY,
        "credentials_used": False,
        "promotion_status": "BLOCKED_OWNER_REVIEW_AND_INDEPENDENT_VALIDATION",
    }


def _validate_report_identity(report: dict[str, Any], expected_sha: str) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("ai6_report_schema_invalid")
    if report.get("commit_sha") != expected_sha:
        raise ValueError("ai6_report_commit_mismatch")
    if report.get("dataset", {}).get("requested_days") != REQUESTED_DAYS:
        raise ValueError("ai6_dataset_window_invalid")


def _validate_report_cost_and_feature_contracts(report: dict[str, Any]) -> None:
    if (
        report.get("execution_assumptions", {}).get("round_trip_cost_bps")
        != ROUND_TRIP_COST_BPS
    ):
        raise ValueError("ai6_cost_contract_invalid")
    if report.get("funding", {}).get("feature_columns") != list(
        EXTERNAL_FEATURE_COLUMNS
    ):
        raise ValueError("ai6_funding_feature_contract_invalid")
    if report.get("funding", {}).get("open_interest_used") is not False:
        raise ValueError("ai6_open_interest_must_be_disabled")
    if report.get("model_usage", {}).get("open_interest_used") is not False:
        raise ValueError("ai6_model_open_interest_must_be_disabled")


def _validate_report_safety(report: dict[str, Any]) -> None:
    expected_safety = {**SAFETY, "credentials_used": False}
    actual_safety = {key: report.get(key) for key in expected_safety}
    if actual_safety != expected_safety:
        raise ValueError("ai6_safety_contract_invalid")


def _validate_report_candidate_row(candidate: object) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError("ai6_candidate_row_invalid")
    identity_checks = (
        candidate.get("symbol") in v4ap.ASSETS,
        candidate.get("timeframe") in v4ap.TIMEFRAMES,
        candidate.get("strategy") in v4ap.STRATEGIES,
        candidate.get("geometry") in {item["name"] for item in v4ap.EXIT_GEOMETRIES},
        candidate.get("round_trip_cost_bps") == ROUND_TRIP_COST_BPS,
        candidate.get("roc_auc_reference") == ROC_AUC_REFERENCE,
        candidate.get("dsr_trial_count") == v4ap.DSR_TRIAL_COUNT,
    )
    if not all(identity_checks):
        raise ValueError("ai6_candidate_contract_invalid")
    expected_decision = v4ap._candidate_qualifies(candidate)
    if candidate.get("qualifies") is not expected_decision:
        raise ValueError("ai6_candidate_decision_invalid")
    return candidate


def _validate_report_candidate_grid(report: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != v4ap.DSR_TRIAL_COUNT:
        raise ValueError("ai6_candidate_grid_cardinality_invalid")
    checked = [_validate_report_candidate_row(candidate) for candidate in candidates]
    candidate_ids = {str(candidate.get("candidate_id")) for candidate in checked}
    if candidate_ids != v4ap._grid_candidate_ids():
        raise ValueError("ai6_candidate_grid_identity_invalid")
    qualifying = [
        str(item["candidate_id"]) for item in checked if v4ap._candidate_qualifies(item)
    ]
    if report.get("qualifying_candidate_ids") != qualifying:
        raise ValueError("ai6_qualifying_candidate_list_invalid")
    return checked


def _validate_report_dsr(report: dict[str, Any]) -> None:
    if report.get("dsr_cohort", {}).get("trial_count") != v4ap.DSR_TRIAL_COUNT:
        raise ValueError("ai6_dsr_trial_count_invalid")


def _validate_report_status(report: dict[str, Any]) -> None:
    status = report.get("status")
    if status not in (
        "COMPLETED_RESEARCH_EVALUATION",
        "BLOCKED_INSUFFICIENT_FUNDING_HISTORY",
    ):
        raise ValueError("ai6_report_status_invalid")
    if (
        status == "BLOCKED_INSUFFICIENT_FUNDING_HISTORY"
        and report.get("economic_results") is not None
    ):
        raise ValueError("ai6_blocked_run_has_economic_results")
    if (
        status == "COMPLETED_RESEARCH_EVALUATION"
        and report.get("economic_results") is None
    ):
        raise ValueError("ai6_completed_run_missing_economic_results")


def _validate_report(report: dict[str, Any], expected_commit_sha: str) -> None:
    _validate_report_identity(report, expected_commit_sha)
    _validate_report_cost_and_feature_contracts(report)
    _validate_report_safety(report)
    _validate_report_candidate_grid(report)
    _validate_report_dsr(report)
    _validate_report_status(report)


def _format_number(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{decimals}f}"


def _render_summary(report: dict[str, Any]) -> str:
    lines = [
        "# AliX AI-6: Funding-only 36-cell Economic Sweep",
        "",
        f"Commit SHA: {report['commit_sha']}",
        f"Status: {report['status']}",
        f"Dataset: {REQUESTED_DAYS} days | 1m Spot OHLCV | "
        f"{report['dataset']['start_utc']} through {report['dataset']['end_utc']}",
        "Assets: BTCUSDT, ETHUSDT | Timeframes: 15m, 1h, 4h",
        "Features: funding_rate_zscore, funding_rate_ma | OI: DISABLED",
        "Primary signals remain the two preregistered V4-AP rules; Funding only filters BUY events.",
        "Model scope: 12 asset/timeframe/strategy fits; each prediction stream is reused across 3 exits.",
        "Forward validation: 70%-85%; final 15% remains untouched.",
        "Training target: fixed 24h net close return after 30 bps round-trip costs.",
        "Execution: next-bar-open spot long; 10 bps fee + 5 bps slippage per side = 30 bps round trip.",
        "",
        "## Acceptance",
        "Trades >= 50; Win Rate >= 40%; Net PF >= 1.10; DSR Z > 0.0.",
        "ROC-AUC: N/A for this run; deferred to keep focus on the preregistered economic sweep. It is not an acceptance gate.",
        "DSR uses the full 36-cell preregistered cohort; trial independence is not verified.",
        "",
        "## All 36 combinations",
        "",
        "| Asset | TF | Strategy | Exit | ROC-AUC | AUC >0.51 | Signals | Trades | Win Rate % | Net PF | DSR Z | Fit status | Decision |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in report["candidates"]:
        pf = (
            "∞"
            if item.get("net_profit_factor_infinite_no_losses")
            else _format_number(item.get("net_profit_factor"))
        )
        auc = _format_number(item.get("roc_auc"))
        above = (
            "N/A"
            if item.get("roc_auc_above_reference") is None
            else "YES"
            if item["roc_auc_above_reference"]
            else "NO"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(item["symbol"]),
                    str(item["timeframe"]),
                    str(item["strategy"]),
                    str(item["geometry"]),
                    auc,
                    above,
                    str(item.get("meta_filtered_signal_count", 0)),
                    str(item["trade_count"]),
                    _format_number(float(item["win_rate"]) * 100.0, 2),
                    pf,
                    _format_number(item.get("dsr_z_score")),
                    str(item.get("meta_model_fit_status", "UNKNOWN")),
                    "PASS" if item["qualifies"] else "NO",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            str(report["conclusion"]),
            "",
            f"Net PF >= 1.10 candidate count: {len(report['net_pf_threshold_candidate_ids'])}",
            f"Fully accepted candidate count: {len(report['qualifying_candidate_ids'])}",
        ]
    )
    if report["status"] == "BLOCKED_INSUFFICIENT_FUNDING_HISTORY":
        lines.extend(
            [
                "",
                "Funding coverage is insufficient. No model was fitted and no economic replay was performed.",
                "This is a data-coverage block, NOT evidence for or against statistical edge.",
                "External Data phase remains UNRESOLVED pending a viable Funding source/window.",
            ]
        )
    elif report["conclusion"] == "NO_NET_PF_EDGE_FOUND_IN_FUNDING_ONLY_FEATURES":
        lines.extend(
            [
                "",
                "External Data phase: CLOSE after this completed negative diagnostic.",
                "Next research track: isolated RL for risk management and position sizing only.",
                "RL must not generate, replace, or modify Primary entry signals.",
            ]
        )
    elif (
        report["conclusion"] == "NET_PF_GATE_MET_BUT_FULL_STATISTICAL_ACCEPTANCE_FAILED"
    ):
        lines.extend(
            [
                "",
                "Profit-factor threshold was reached in at least one cell, but no cell passed all pre-registered statistical gates.",
                "This is not a validated winner and does not authorize promotion or trading.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "One or more cells passed all economic and statistical gates.",
                "This remains a research candidate only; independent owner review is required.",
            ]
        )
    lines.extend(
        [
            "",
            "## Safety / promotion",
            "",
            "Champion=null | order_permission=false | trading_authorization=NO-GO",
            "private_api_calls=0 | real_order_attempts=0 | credentials_used=false",
            "No order session, paper session, dry-run session, or exchange account access was used.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_rows(
    path: Path, rows: list[dict[str, Any]], fallback_columns: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else list(fallback_columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, allow_nan=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _write_outputs(
    report: dict[str, Any],
    output_report: Path,
    output_trades_csv: Path,
    output_decisions_csv: Path,
    output_summary: Path,
    trades: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> None:
    _validate_report(report, str(report["commit_sha"]))
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_rows(
        output_trades_csv,
        trades,
        ("candidate_id", "symbol", "timeframe", "strategy", "geometry"),
    )
    _write_rows(
        output_decisions_csv,
        decisions,
        ("symbol", "timeframe", "strategy", "event_index", "probability", "accepted"),
    )
    summary = _render_summary(report)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(summary, encoding="utf-8")
    print(summary, end="")


def _load_funding_inputs(
    external_dir: Path,
    commit_sha: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, DataFrame], dict[str, dict[str, Any]], bool]:
    funding_frames: dict[str, DataFrame] = {}
    funding_provenance: dict[str, dict[str, Any]] = {}
    funding_ready = True
    for symbol in v4ap.ASSETS:
        frame, manifest, ready = _read_funding_artifact(
            external_dir, symbol, commit_sha, start, end
        )
        funding_frames[symbol] = frame
        funding_provenance[symbol] = manifest
        funding_ready = funding_ready and ready
    return funding_frames, funding_provenance, funding_ready


def _blocked_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for symbol in v4ap.ASSETS:
        for timeframe in v4ap.TIMEFRAMES:
            for strategy in v4ap.STRATEGIES:
                for geometry in v4ap.EXIT_GEOMETRIES:
                    candidates.append(
                        _placeholder_candidate(
                            symbol,
                            timeframe,
                            strategy,
                            geometry,
                            status="NOT_RUN_INSUFFICIENT_FUNDING_HISTORY",
                        )
                    )
    return candidates


def _build_gap_ranges(asset_provenance: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(gap["start_utc"]), str(gap["end_utc"]))
        for gap in asset_provenance["missing_gap_ranges"]
    )


def _annotate_sweep_candidate(
    candidate: dict[str, Any],
    *,
    group_result: dict[str, Any],
    strategy_diagnostics: dict[str, Any],
    filtered_signal: Series,
    train_end: int,
    forward_end: int,
) -> None:
    group_auc = group_result["roc_auc"]
    candidate.update(
        {
            "roc_auc": group_auc,
            "roc_auc_reference": ROC_AUC_REFERENCE,
            "roc_auc_status": group_result["roc_auc_status"],
            "roc_auc_above_reference": (
                None if group_auc is None else bool(group_auc > ROC_AUC_REFERENCE)
            ),
            "unfiltered_primary_signal_count": int(
                strategy_diagnostics["primary_signal_count"]
            ),
            "meta_filtered_signal_count": int(filtered_signal.sum()),
            "meta_filter_acceptance_rate": group_result["acceptance_rate"],
            "meta_model_fit_status": group_result["fit_status"],
            "meta_model_fit_rows": group_result["fit_rows"],
            "meta_model_calibration_rows": group_result["calibration_rows"],
            "validation_event_count": group_result["validation_event_count"],
            "validation_labeled_event_count": group_result[
                "validation_labeled_event_count"
            ],
            "meta_threshold": group_result["threshold"],
            "validation_start_index": train_end,
            "validation_end_exclusive": forward_end,
        }
    )


def _run_primary_strategy_sweep(
    evaluation_bars: DataFrame,
    *,
    symbol: str,
    timeframe: str,
    strategy: str,
    signal_generator: Any,
    funding_diagnostics: dict[str, Any],
    execution: Any,
    documented_gap_ranges: tuple[tuple[str, str], ...],
    train_end: int,
    forward_end: int,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    primary_signal, strategy_diagnostics = signal_generator(evaluation_bars)
    primary_features = build_v4aq_meta_features(
        evaluation_bars, funding_diagnostics["features"], strategy=strategy
    )
    filtered_signal, group_result, decision_rows = _evaluate_primary_group(
        evaluation_bars,
        primary_features,
        primary_signal,
        strategy=strategy,
        timeframe=timeframe,
        train_end=train_end,
        forward_end=forward_end,
    )
    model_group = {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy,
        "funding_feature_diagnostics": funding_diagnostics["diagnostics"],
        **group_result,
    }
    candidate_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    decision_records = [
        {"symbol": symbol, "timeframe": timeframe, "strategy": strategy, **row}
        for row in decision_rows
    ]
    for geometry in v4ap.EXIT_GEOMETRIES:
        run_diagnostics = {
            **strategy_diagnostics,
            "primary_signal_count": int(filtered_signal.sum()),
        }
        candidate, candidate_trades = v4ap._run_candidate(
            evaluation_bars,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            geometry=geometry,
            signal=filtered_signal,
            strategy_diagnostics=run_diagnostics,
            execution=execution,
            documented_gap_ranges=documented_gap_ranges,
        )
        _annotate_sweep_candidate(
            candidate,
            group_result=group_result,
            strategy_diagnostics=strategy_diagnostics,
            filtered_signal=filtered_signal,
            train_end=train_end,
            forward_end=forward_end,
        )
        candidate_rows.append(candidate)
        trade_rows.extend(candidate_trades)
    return candidate_rows, trade_rows, decision_records, model_group


def _prepare_timeframe_features(
    evaluation_bars: DataFrame,
    funding_frame: DataFrame,
    symbol: str,
    timeframe: str,
) -> tuple[DataFrame, dict[str, Any]]:
    features, diagnostics = build_external_market_features(
        evaluation_bars,
        funding_frame,
        symbol=symbol,
        timeframe=timeframe,
    )
    return features, {"features": features, "diagnostics": diagnostics}


def _run_timeframe_sweep(
    minute_data: DataFrame,
    funding_frame: DataFrame,
    *,
    symbol: str,
    timeframe: str,
    signal_generators: dict[str, Any],
    execution: Any,
    documented_gap_ranges: tuple[tuple[str, str], ...],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    full_bars = resample_ohlcv(minute_data, timeframe)
    train_end = int(len(full_bars) * TRAIN_FRACTION)
    forward_end = int(len(full_bars) * FORWARD_END_FRACTION)
    horizon_bars = v4ap.HORIZON_HOURS * 60 // v4ap.TIMEFRAME_MINUTES[timeframe]
    if train_end <= horizon_bars + 5 or forward_end <= train_end + horizon_bars:
        raise ValueError(f"ai6_resampled_window_too_short:{symbol}:{timeframe}")
    evaluation_bars = full_bars.iloc[:forward_end].copy().reset_index(drop=True)
    funding_features, funding_feature_diagnostics = _prepare_timeframe_features(
        evaluation_bars, funding_frame, symbol, timeframe
    )
    strategy_context = {
        "features": funding_features,
        "diagnostics": funding_feature_diagnostics["diagnostics"],
    }
    candidate_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    model_groups: list[dict[str, Any]] = []
    for strategy in v4ap.STRATEGIES:
        candidates, trades, decisions, model_group = _run_primary_strategy_sweep(
            evaluation_bars,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            signal_generator=signal_generators[strategy],
            funding_diagnostics=strategy_context,
            execution=execution,
            documented_gap_ranges=documented_gap_ranges,
            train_end=train_end,
            forward_end=forward_end,
        )
        candidate_rows.extend(candidates)
        trade_rows.extend(trades)
        decision_rows.extend(decisions)
        model_groups.append(model_group)
    return candidate_rows, trade_rows, decision_rows, model_groups


def _run_asset_sweep(
    input_dir: Path,
    funding_frame: DataFrame,
    *,
    symbol: str,
    commit_sha: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    signal_generators: dict[str, Any],
    execution: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    minute_data, asset_provenance = v4ap._load_market_data(
        input_dir,
        symbol=symbol,
        commit_sha=commit_sha,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    documented_gap_ranges = _build_gap_ranges(asset_provenance)
    candidate_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    model_groups: list[dict[str, Any]] = []
    for timeframe in v4ap.TIMEFRAMES:
        candidates, trades, decisions, groups = _run_timeframe_sweep(
            minute_data,
            funding_frame,
            symbol=symbol,
            timeframe=timeframe,
            signal_generators=signal_generators,
            execution=execution,
            documented_gap_ranges=documented_gap_ranges,
        )
        candidate_rows.extend(candidates)
        trade_rows.extend(trades)
        decision_rows.extend(decisions)
        model_groups.extend(groups)
    return (
        candidate_rows,
        trade_rows,
        decision_rows,
        model_groups,
        asset_provenance,
    )


def _run_complete_sweep(
    input_dir: Path,
    funding_frames: dict[str, DataFrame],
    commit_sha: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    execution = v4ap._execution_config()
    signal_generators = {
        v4ap.STRATEGIES[0]: v4ap.generate_mean_reversion_signal,
        v4ap.STRATEGIES[1]: v4ap.generate_donchian_breakout_signal,
    }
    candidates: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    market_provenance: dict[str, dict[str, Any]] = {}
    model_groups: list[dict[str, Any]] = []
    for symbol in v4ap.ASSETS:
        asset_candidates, asset_trades, asset_decisions, asset_groups, provenance = (
            _run_asset_sweep(
                input_dir,
                funding_frames[symbol],
                symbol=symbol,
                commit_sha=commit_sha,
                start_utc=start,
                end_utc=end,
                signal_generators=signal_generators,
                execution=execution,
            )
        )
        candidates.extend(asset_candidates)
        trades.extend(asset_trades)
        decisions.extend(asset_decisions)
        model_groups.extend(asset_groups)
        market_provenance[symbol] = provenance
    if len(candidates) != v4ap.DSR_TRIAL_COUNT:
        raise ValueError("ai6_sweep_candidate_count_invalid")
    return candidates, trades, decisions, market_provenance, model_groups


def run_v4aq_ai6(
    input_dir: Path,
    external_dir: Path,
    output_report: Path,
    output_trades_csv: Path,
    output_decisions_csv: Path,
    output_summary: Path,
    *,
    commit_sha: str,
    start_utc: str,
    end_utc: str,
) -> dict[str, Any]:
    """Run the Funding-only 36-cell sweep after exact-window coverage verification."""
    _validate_commit_sha(commit_sha)
    start, end = v4ap._validate_request(commit_sha, start_utc, end_utc)
    funding_frames, funding_provenance, funding_ready = _load_funding_inputs(
        external_dir, commit_sha, start, end
    )
    market_provenance: dict[str, dict[str, Any]] = {}
    model_groups: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    if funding_ready:
        (
            candidates,
            trades,
            decisions,
            market_provenance,
            model_groups,
        ) = _run_complete_sweep(input_dir, funding_frames, commit_sha, start, end)
        status = "COMPLETED_RESEARCH_EVALUATION"
    else:
        candidates = _blocked_candidates()
        status = "BLOCKED_INSUFFICIENT_FUNDING_HISTORY"
    dsr_cohort = v4ap._attach_dsr_to_cohort(candidates)
    for candidate in candidates:
        candidate["qualifies"] = v4ap._candidate_qualifies(candidate)
    report = _build_report(
        commit_sha=commit_sha,
        start_utc=start,
        end_utc=end,
        market_provenance=market_provenance,
        funding_provenance=funding_provenance,
        candidates=candidates,
        dsr_cohort=dsr_cohort,
        model_groups=model_groups,
        status=status,
    )
    _write_outputs(
        report,
        output_report,
        output_trades_csv,
        output_decisions_csv,
        output_summary,
        trades,
        decisions,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-trades-csv", type=Path, required=True)
    parser.add_argument("--output-decisions-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--end-utc", required=True)
    args = parser.parse_args()
    run_v4aq_ai6(
        args.input_dir,
        args.external_dir,
        args.output_report,
        args.output_trades_csv,
        args.output_decisions_csv,
        args.output_summary,
        commit_sha=args.commit_sha,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
