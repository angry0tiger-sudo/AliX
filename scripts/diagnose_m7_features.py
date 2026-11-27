"""Diagnose M7 feature availability on the immutable PRE-HACK panel."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.run_m7_cross_sectional import (
    BTC_SYMBOL,
    EXPECTED_ROWS,
    _beta_24h,
    _log_returns,
    _peer_median_return,
    _relative_volume_shock,
    load_locked_data,
    sha256_file,
)

EXPECTED_SYMBOLS = (
    "ETHUSDT",
    "BNBUSDT",
    "ADAUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "LINKUSDT",
    "DOTUSDT",
)
EXPECTED_TRIALS: Mapping[str, tuple[str, ...]] = {
    "NOBITEX-M7-5PAIR-RIDGE-TOP2-02": EXPECTED_SYMBOLS[:5],
    "NOBITEX-M7-9PAIR-RIDGE-TOP2-01": EXPECTED_SYMBOLS,
}
DECISION_INDICES = tuple(range(24, EXPECTED_ROWS - 5, 4))
BETA_REJECTION_REASONS = (
    "asset_return_window_nonfinite",
    "btc_return_window_nonfinite",
    "btc_variance_nonfinite_or_at_or_below_1e-12",
    "beta_output_nonfinite",
    "beta_helper_returned_none",
)
VOLUME_REJECTION_REASONS = (
    "prior_24h_median_volume_nonfinite",
    "prior_24h_median_volume_nonpositive",
    "current_completed_hour_volume_invalid",
    "volume_shock_output_nonfinite",
    "volume_helper_returned_none",
)

def _rate_pct(count: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * count / denominator, 6)

def inspect_beta_feature(
    asset_returns: np.ndarray,
    btc_returns: np.ndarray,
    decision_index: int,
) -> tuple[float | None, str | None]:
    """Apply the production beta helper and identify its first rejection cause."""
    asset_window = asset_returns[decision_index - 23 : decision_index + 1]
    btc_window = btc_returns[decision_index - 23 : decision_index + 1]
    if not np.isfinite(asset_window).all():
        return None, "asset_return_window_nonfinite"
    if not np.isfinite(btc_window).all():
        return None, "btc_return_window_nonfinite"
    btc_variance = float(np.var(btc_window, ddof=1))
    if not math.isfinite(btc_variance) or btc_variance <= 1e-12:
        return None, "btc_variance_nonfinite_or_at_or_below_1e-12"
    value = _beta_24h(asset_returns, btc_returns, decision_index)
    if value is None:
        return None, "beta_helper_returned_none"
    if not math.isfinite(value):
        return None, "beta_output_nonfinite"
    return value, None

def inspect_volume_feature(
    volume: np.ndarray,
    decision_index: int,
) -> tuple[float | None, str | None]:
    """Apply the production relative-volume helper and identify rejection cause."""
    prior_volume = volume[decision_index - 24 : decision_index]
    median_volume = float(np.median(prior_volume))
    current_volume = float(volume[decision_index])
    if not math.isfinite(median_volume):
        return None, "prior_24h_median_volume_nonfinite"
    if median_volume <= 0.0:
        return None, "prior_24h_median_volume_nonpositive"
    if not math.isfinite(current_volume) or current_volume < 0.0:
        return None, "current_completed_hour_volume_invalid"
    value = _relative_volume_shock(volume, decision_index)
    if value is None:
        return None, "volume_helper_returned_none"
    if not math.isfinite(value):
        return None, "volume_shock_output_nonfinite"
    return value, None

def _feature_summary(
    valid_count: int,
    total_count: int,
    reasons: Counter[str],
    known_reasons: Sequence[str],
) -> dict[str, Any]:
    rejected = total_count - valid_count
    return {
        "valid_rows": valid_count,
        "rejected_rows": rejected,
        "rejection_rate_pct": _rate_pct(rejected, total_count),
        "valid_rate_pct": _rate_pct(valid_count, total_count),
        "rejection_reasons": {
            reason: reasons.get(reason, 0) for reason in known_reasons
        },
    }

def _collect_symbol_feature_flags(
    asset_returns: np.ndarray,
    volume: np.ndarray,
    btc_returns: np.ndarray,
    decision_indices: Sequence[int],
) -> tuple[list[bool], list[bool], Counter[str], Counter[str]]:
    beta_flags: list[bool] = []
    volume_flags: list[bool] = []
    beta_reasons: Counter[str] = Counter()
    volume_reasons: Counter[str] = Counter()
    for index in decision_indices:
        beta_value, beta_reason = inspect_beta_feature(
            asset_returns, btc_returns, index
        )
        volume_value, volume_reason = inspect_volume_feature(volume, index)
        beta_flags.append(beta_reason is None and beta_value is not None)
        volume_flags.append(volume_reason is None and volume_value is not None)
        if beta_reason is not None:
            beta_reasons[beta_reason] += 1
        if volume_reason is not None:
            volume_reasons[volume_reason] += 1
    return beta_flags, volume_flags, beta_reasons, volume_reasons

def _flag_overlap_summary(
    beta_flags: Sequence[bool], volume_flags: Sequence[bool]
) -> dict[str, int]:
    pairs = list(zip(beta_flags, volume_flags, strict=True))
    return {
        "both_features_rejected_rows": sum(
            not beta_ok and not volume_ok for beta_ok, volume_ok in pairs
        ),
        "beta_only_rejected_rows": sum(
            not beta_ok and volume_ok for beta_ok, volume_ok in pairs
        ),
        "volume_only_rejected_rows": sum(
            beta_ok and not volume_ok for beta_ok, volume_ok in pairs
        ),
    }

def _inspect_symbol(
    symbol: str,
    frame: pd.DataFrame,
    btc_returns: np.ndarray,
    decision_indices: Sequence[int],
) -> dict[str, Any]:
    asset_returns = _log_returns(frame["close"].to_numpy(dtype=float))
    volume = frame["volume"].to_numpy(dtype=float)
    beta_flags, volume_flags, beta_reasons, volume_reasons = (
        _collect_symbol_feature_flags(
            asset_returns, volume, btc_returns, decision_indices
        )
    )
    total = len(decision_indices)
    overlap = _flag_overlap_summary(beta_flags, volume_flags)
    return {
        "report": {
            "symbol": symbol,
            "decision_rows": total,
            "beta_24h_to_btc": _feature_summary(
                sum(beta_flags), total, beta_reasons, BETA_REJECTION_REASONS
            ),
            "relative_volume_shock_24h": _feature_summary(
                sum(volume_flags), total, volume_reasons, VOLUME_REJECTION_REASONS
            ),
            "suspect_feature_overlap": overlap,
        },
        "_beta_flags": beta_flags,
        "_volume_flags": volume_flags,
    }

def _other_features_valid(
    symbol: str,
    decision_index: int,
    symbols: Sequence[str],
    btc_close: np.ndarray,
    close_by_symbol: Mapping[str, np.ndarray],
    open_by_symbol: Mapping[str, np.ndarray],
) -> bool:
    btc_return_4h = math.log(btc_close[decision_index] / btc_close[decision_index - 4])
    peer_median = _peer_median_return(close_by_symbol, symbols, symbol, decision_index)
    entry_open = float(open_by_symbol[symbol][decision_index + 1])
    exit_open = float(open_by_symbol[symbol][decision_index + 5])
    target_log_return = math.log(exit_open / entry_open)
    target_simple_return = exit_open / entry_open - 1.0
    return all(
        math.isfinite(value)
        for value in (
            btc_return_4h,
            peer_median,
            target_log_return,
            target_simple_return,
        )
    )

def _candidate_panel_summary(
    trial_id: str,
    symbols: Sequence[str],
    states: Mapping[str, Mapping[str, Any]],
    decision_indices: Sequence[int],
    data: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    btc_close = data[BTC_SYMBOL]["close"].to_numpy(dtype=float)
    close_by_symbol = {
        symbol: data[symbol]["close"].to_numpy(dtype=float) for symbol in symbols
    }
    open_by_symbol = {
        symbol: data[symbol]["open"].to_numpy(dtype=float) for symbol in symbols
    }

    total_rows = len(symbols) * len(decision_indices)
    beta_rejected = 0
    volume_rejected = 0
    both_rejected = 0
    beta_only_rejected = 0
    volume_only_rejected = 0
    other_feature_rejected = 0
    accepted_rows = 0

    for position, decision_index in enumerate(decision_indices):
        for symbol in symbols:
            beta_valid = bool(states[symbol]["_beta_flags"][position])
            volume_valid = bool(states[symbol]["_volume_flags"][position])
            other_valid = _other_features_valid(
                symbol,
                decision_index,
                symbols,
                btc_close,
                close_by_symbol,
                open_by_symbol,
            )
            beta_rejected += int(not beta_valid)
            volume_rejected += int(not volume_valid)
            both_rejected += int(not beta_valid and not volume_valid)
            beta_only_rejected += int(not beta_valid and volume_valid)
            volume_only_rejected += int(beta_valid and not volume_valid)
            other_feature_rejected += int(not other_valid)
            accepted_rows += int(beta_valid and volume_valid and other_valid)

    rejected_rows = total_rows - accepted_rows
    return {
        "trial_id": trial_id,
        "symbols": list(symbols),
        "decision_timestamps": len(decision_indices),
        "candidate_rows_before_feature_qualification": total_rows,
        "beta_rejected_rows": beta_rejected,
        "beta_rejection_rate_pct": _rate_pct(beta_rejected, total_rows),
        "volume_rejected_rows": volume_rejected,
        "volume_rejection_rate_pct": _rate_pct(volume_rejected, total_rows),
        "both_suspect_features_rejected_rows": both_rejected,
        "beta_only_rejected_rows": beta_only_rejected,
        "volume_only_rejected_rows": volume_only_rejected,
        "other_feature_or_target_invalid_rows": other_feature_rejected,
        "accepted_panel_rows": accepted_rows,
        "total_panel_rejected_rows": rejected_rows,
        "total_panel_rejection_rate_pct": _rate_pct(rejected_rows, total_rows),
    }

def _validate_g2_preregistration(config: Mapping[str, Any]) -> str:
    preregistration = config["preregistration"]
    if preregistration.get("revision") != "M7-CS-G2.0":
        raise ValueError("m7_diagnostic_requires_g2_revision")
    if preregistration.get("status") != "PREREGISTERED_LOCKED":
        raise ValueError("m7_diagnostic_requires_locked_preregistration")
    if preregistration.get("logical_trial_count") != 2:
        raise ValueError("m7_diagnostic_trial_count_mismatch")
    return str(preregistration["revision"])

def _validate_g2_universes(config: Mapping[str, Any]) -> None:
    if tuple(config["data"]["candidate_symbol_priority"]) != EXPECTED_SYMBOLS:
        raise ValueError("m7_diagnostic_unexpected_symbol_priority")
    registered_trials = {
        str(trial["trial_id"]): tuple(trial["symbols"]) for trial in config["trials"]
    }
    if registered_trials != dict(EXPECTED_TRIALS):
        raise ValueError("m7_diagnostic_trial_universe_mismatch")

def _read_qualified_manifest(data_dir: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = data_dir / "m7_data_qualification.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qualified = tuple(manifest.get("qualified_symbols", ()))
    if manifest.get("data_scope") != "PRE_HACK_ONLY":
        raise ValueError("m7_diagnostic_manifest_scope_mismatch")
    if len(qualified) != len(EXPECTED_SYMBOLS):
        raise ValueError("m7_diagnostic_manifest_symbol_count_mismatch")
    if set(qualified) != set(EXPECTED_SYMBOLS):
        raise ValueError("m7_diagnostic_manifest_universe_mismatch")
    return manifest_path, manifest

def _load_verified_diagnostic_data(
    data_dir: Path, btc_csv: Path
) -> dict[str, pd.DataFrame]:
    data = load_locked_data(data_dir, btc_csv)
    if set(data) != {BTC_SYMBOL, *EXPECTED_SYMBOLS}:
        raise ValueError("m7_diagnostic_loaded_universe_mismatch")
    if len(data[BTC_SYMBOL]["timestamp"]) != EXPECTED_ROWS:
        raise ValueError("m7_diagnostic_btc_row_count_mismatch")
    return data

def diagnose_m7_features(
    data_dir: Path,
    btc_csv: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Read the locked nine-pair PRE-HACK dataset and report feature availability."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    revision = _validate_g2_preregistration(config)
    _validate_g2_universes(config)
    manifest_path, _manifest = _read_qualified_manifest(data_dir)
    data = _load_verified_diagnostic_data(data_dir, btc_csv)
    btc_returns = _log_returns(data[BTC_SYMBOL]["close"].to_numpy(dtype=float))
    symbol_states = {
        symbol: _inspect_symbol(symbol, data[symbol], btc_returns, DECISION_INDICES)
        for symbol in EXPECTED_SYMBOLS
    }
    per_symbol = {symbol: state["report"] for symbol, state in symbol_states.items()}
    candidate_reports = {
        trial_id: _candidate_panel_summary(
            trial_id, symbols, symbol_states, DECISION_INDICES, data
        )
        for trial_id, symbols in EXPECTED_TRIALS.items()
    }
    return {
        "schema_version": "alix.m7.feature-diagnostic.v1",
        "phase": "M7",
        "diagnostic": "FEATURE_AVAILABILITY_PREHACK_ONLY",
        "preregistration_revision": revision,
        "preregistration_sha256": sha256_file(config_path),
        "qualification_manifest_sha256": sha256_file(manifest_path),
        "btc_source_sha256": sha256_file(btc_csv),
        "source_commit_sha": os.environ.get("GITHUB_SHA", "UNKNOWN"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "UNKNOWN"),
        "data_scope": "PRE_HACK_ONLY",
        "timeframe": "1h",
        "decision_cadence": "4h",
        "expected_rows_per_symbol": EXPECTED_ROWS,
        "decision_rows_per_symbol": len(DECISION_INDICES),
        "decision_index_start": DECISION_INDICES[0],
        "decision_index_end": DECISION_INDICES[-1],
        "symbols": per_symbol,
        "candidate_panels": candidate_reports,
        "interpretation": {
            "feature_rejection_counts_are_mutually_exclusive_per_feature": True,
            "beta_and_volume_rejections_can_overlap": True,
            "panel_row_is_accepted_only_when_both_suspect_features_and_other_registered_fields_are_finite": True,
            "no_imputation_or_feature_relaxation": True,
        },
        "safety": {
            "new_data_acquisition": False,
            "model_fitting": False,
            "economic_backtest": False,
            "holdout_bytes_read": False,
            "posthack_data_read": False,
            "private_api_calls": 0,
            "paper_orders": 0,
            "real_order_attempts": 0,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "local_execution": False,
        },
    }

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose M7 feature availability using immutable PRE-HACK data."
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
        "--report-out",
        type=Path,
        default=Path(".artifacts/m7/m7_feature_diagnostic.json"),
    )
    args = parser.parse_args()

    report = diagnose_m7_features(args.data_dir, args.btc_csv, args.config)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
