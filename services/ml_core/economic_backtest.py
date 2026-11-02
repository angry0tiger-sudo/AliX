"""Deterministic, non-training economic replay for persisted V4-H signals."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from services.ml_core.adaptive_targets import ATR_PERIOD, calculate_atr

V4H_SOURCE_SCHEMA = "alix.ml-core.v4h-tuning-geometry.v1"
V4H_SOURCE_SHA = "e34861c4873b7cc2170aaaf6c79ed9b464df0a9a"
V4H_SOURCE_RUN_ID = 41293168191
V4H_SOURCE_ARTIFACT_ID = 13080269872
V4H_ACCEPTED_GEOMETRY = "V4H_Tighter_SL"
V4H_MIN_HOLD_PRECISION = 0.40
V4H_MIN_HOLD_RECALL = 0.20
LABELS = ("BUY", "SELL", "HOLD")
NO_ENTRY_SIGNAL = "NO_ENTRY"
TIMEFRAME_MINUTES = {"15m": 15, "30m": 30, "1h": 60, "4h": 240}
OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")
MAX_DRAWDOWN_ACCOUNTING_TOLERANCE = 1e-10


@dataclass(frozen=True)
class ExecutionConfig:
    """Explicit one-times-equity execution assumptions for replay."""

    initial_equity: float = 10_000.0
    fee_bps: float = 25.0
    slippage_bps: float = 20.0
    position_fraction: float = 1.0
    maximum_drawdown: float = 0.15


@dataclass(frozen=True)
class TradeManagementConfig:
    """Triple-Barrier geometry, entry confidence, and optional causal trailing stop."""

    atr_period: int = ATR_PERIOD
    take_profit_atr_multiplier: float = 2.0
    stop_loss_atr_multiplier: float = 1.5
    horizon_candles: int = 12
    confidence_threshold: float = 0.55
    trailing_stop_enabled: bool = False
    trailing_activation_atr: float = 1.0
    trailing_step_atr: float = 1.0
    time_exit_candles: int | None = None
    time_exit_mfe_threshold_atr: float = 0.5


@dataclass(frozen=True)
class OpenPosition:
    """One non-pyramided spot long with fixed ATR-derived exit barriers."""

    direction: int
    entry_timestamp: str
    entry_reference_price: float
    entry_fill_price: float
    quantity: float
    equity_at_entry: float
    entry_fee: float
    entry_slippage_cost: float
    atr_value: float
    take_profit_price: float
    stop_loss_price: float
    entry_bar_index: int
    entry_confidence: float
    atr_reference_timestamp: str
    initial_stop_loss_price: float
    high_water_mark: float
    trailing_stop_active: bool
    trailing_stop_steps: int
    time_exit_candles: int | None
    time_exit_mfe_threshold_atr: float
    time_exit_evaluated: bool
    time_exit_pending: bool
    max_favorable_excursion_atr: float
    max_adverse_excursion_atr: float


def validate_execution_config(config: ExecutionConfig) -> None:
    """Reject invalid or leveraged simulation assumptions."""
    values = (
        config.initial_equity,
        config.fee_bps,
        config.slippage_bps,
        config.position_fraction,
        config.maximum_drawdown,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("economic_config_values_must_be_finite")
    if config.initial_equity <= 0:
        raise ValueError("initial_equity_must_be_positive")
    if config.fee_bps < 0 or config.slippage_bps < 0:
        raise ValueError("execution_costs_must_be_non_negative")
    if not 0 < config.position_fraction <= 1:
        raise ValueError("position_fraction_must_be_in_(0,1]")
    if not 0 < config.maximum_drawdown < 1:
        raise ValueError("maximum_drawdown_must_be_in_(0,1)")


def _require_positive_integer(value: Any, error_code: str) -> None:
    """Require a positive integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(error_code)


def _validate_trade_barrier_multipliers(config: TradeManagementConfig) -> None:
    """Validate finite, positive ATR barrier multipliers."""
    values = (config.take_profit_atr_multiplier, config.stop_loss_atr_multiplier)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("economic_trade_management_values_must_be_finite")
    if config.take_profit_atr_multiplier <= 0 or config.stop_loss_atr_multiplier <= 0:
        raise ValueError("economic_atr_multipliers_must_be_positive")


def _validate_trailing_stop_settings(config: TradeManagementConfig) -> None:
    """Reject malformed trailing-stop enablement, activation, or step settings."""
    if not isinstance(config.trailing_stop_enabled, bool):
        raise ValueError("economic_trailing_stop_enabled_invalid")
    values = (config.trailing_activation_atr, config.trailing_step_atr)
    if any(isinstance(value, bool) for value in values):
        raise ValueError("economic_trailing_stop_settings_invalid")
    try:
        valid = all(np.isfinite(float(value)) and float(value) > 0 for value in values)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("economic_trailing_stop_settings_invalid")


def _validate_time_exit_settings(config: TradeManagementConfig) -> None:
    """Reject malformed dynamic time-exit settings."""
    if config.time_exit_candles is not None:
        _require_positive_integer(
            config.time_exit_candles, "economic_time_exit_candles_invalid"
        )
        if config.time_exit_candles >= config.horizon_candles:
            raise ValueError("economic_time_exit_must_precede_vertical_horizon")
    threshold = config.time_exit_mfe_threshold_atr
    if isinstance(threshold, bool):
        raise ValueError("economic_time_exit_mfe_threshold_invalid")
    try:
        valid = np.isfinite(float(threshold)) and float(threshold) > 0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("economic_time_exit_mfe_threshold_invalid")


def validate_trade_management_config(config: TradeManagementConfig) -> None:
    """Reject invalid ATR, barrier, horizon, confidence, and exit settings."""
    _require_positive_integer(config.atr_period, "economic_atr_period_invalid")
    _require_positive_integer(
        config.horizon_candles, "economic_horizon_candles_invalid"
    )
    _validate_trade_barrier_multipliers(config)
    _validate_confidence_threshold(config.confidence_threshold)
    _validate_trailing_stop_settings(config)
    _validate_time_exit_settings(config)


def _source_report_metadata_is_approved(report: dict[str, Any]) -> bool:
    """Check immutable V4-H identity and training contract fields."""
    acceptance = report.get("acceptance_policy")
    return (
        report.get("schema_version") == V4H_SOURCE_SCHEMA
        and report.get("commit_sha") == V4H_SOURCE_SHA
        and report.get("diagnostic_scope") == "CLASSIFICATION_ONLY"
        and report.get("economic_evaluation_performed") is False
        and report.get("trained_models") == 96
        and report.get("experiment_count") == 48
        and isinstance(acceptance, dict)
        and acceptance.get("status") == "MET"
        and acceptance.get("strict_acceptance_count") == 13
    )


def _source_report_safety_is_closed(report: dict[str, Any]) -> bool:
    """Verify source report safety state before any economic replay."""
    safety = report.get("safety")
    return (
        isinstance(safety, dict)
        and safety.get("Champion") is None
        and safety.get("order_permission") is False
        and safety.get("trading_authorization") == "NO-GO"
        and safety.get("private_api_calls") == 0
        and safety.get("real_order_attempts") == 0
    )


def validate_source_report(report: dict[str, Any]) -> None:
    """Require the exact approved V4-H report and closed safety state."""
    if not _source_report_metadata_is_approved(report):
        raise ValueError("v4h_source_report_contract_invalid")
    if not _source_report_safety_is_closed(report):
        raise ValueError("v4h_source_report_contract_invalid")


def _candidate_forward_metrics(
    variant: dict[str, Any],
) -> tuple[dict[str, Any] | None, float | None, float | None]:
    """Return persisted Forward HOLD metrics, if the candidate has them."""
    threshold_report = variant.get("threshold_optimization") or {}
    forward = threshold_report.get("post_calibration_validation")
    if not isinstance(forward, dict):
        return None, None, None
    hold = forward.get("hold_metrics") or {}
    precision = hold.get("precision")
    recall = hold.get("recall")
    if precision is None or recall is None:
        return forward, None, None
    return forward, float(precision), float(recall)


def _candidate_passes_hold_acceptance(precision: float, recall: float) -> bool:
    """Apply the approved strict V4-H precision and recall floors."""
    return precision >= V4H_MIN_HOLD_PRECISION and recall >= V4H_MIN_HOLD_RECALL


def _candidate_replay_inputs(
    variant: dict[str, Any], forward: dict[str, Any]
) -> tuple[float, int, list[dict[str, Any]]]:
    """Validate required threshold and persisted validation probability payload."""
    threshold_report = variant.get("threshold_optimization") or {}
    threshold = threshold_report.get("selected_threshold")
    start_index = forward.get("start_row_index")
    parameters = (variant.get("hyperparameter_tuning") or {}).get("selected_parameters")
    raw_records = variant.get("validation_probabilities")
    if threshold is None or start_index is None:
        raise ValueError("accepted_v4h_candidate_replay_data_missing")
    if not isinstance(parameters, dict) or not isinstance(raw_records, list):
        raise ValueError("accepted_v4h_candidate_replay_data_missing")
    return float(threshold), int(start_index), raw_records


def _select_forward_probability_records(
    raw_records: list[dict[str, Any]],
    start_index: int,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Select only post-calibration Forward rows and verify their exact count."""
    records = [item for item in raw_records if int(item["row_index"]) >= start_index]
    if len(records) != expected_count:
        raise ValueError("forward_probability_row_count_mismatch")
    if not records:
        raise ValueError("forward_probability_rows_empty")
    return records


def _candidate_forward_records(
    variant: dict[str, Any], forward: dict[str, Any]
) -> tuple[float, int, list[dict[str, Any]]]:
    """Return the threshold, first Forward row index, and persisted probabilities."""
    threshold, start_index, raw_records = _candidate_replay_inputs(variant, forward)
    records = _select_forward_probability_records(
        raw_records, start_index, int(forward.get("row_count", -1))
    )
    return threshold, start_index, records


def _candidate_identifier(experiment: dict[str, Any], policy: str) -> str:
    """Create a stable ID for a single V4-H model variant."""
    return "|".join(
        (
            str(experiment["scenario"]),
            str(experiment["pair"]),
            str(experiment["timeframe"]),
            str(experiment["feature_profile"]),
            policy,
        )
    )


def _extract_candidate_variant(
    experiment: dict[str, Any], policy: str, variant: dict[str, Any]
) -> dict[str, Any] | None:
    """Build a candidate record only when both Forward HOLD floors are met."""
    forward, precision, recall = _candidate_forward_metrics(variant)
    if forward is None or precision is None or recall is None:
        return None
    if not _candidate_passes_hold_acceptance(precision, recall):
        return None
    threshold, start_index, forward_records = _candidate_forward_records(
        variant, forward
    )
    return {
        "candidate_id": _candidate_identifier(experiment, policy),
        "scenario": experiment["scenario"],
        "pair": experiment["pair"],
        "timeframe": experiment["timeframe"],
        "feature_profile": experiment["feature_profile"],
        "weighting_policy": policy,
        "selected_parameters": (variant.get("hyperparameter_tuning") or {})[
            "selected_parameters"
        ],
        "hold_precision": precision,
        "hold_recall": recall,
        "selected_hold_threshold": threshold,
        "forward_start_row_index": start_index,
        "forward_row_count": len(forward_records),
        "probability_records": forward_records,
    }


def _extract_experiment_candidates(
    experiment: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract every accepted class-weight variant from one experiment."""
    if experiment.get("scenario") != V4H_ACCEPTED_GEOMETRY:
        return []
    candidates: list[dict[str, Any]] = []
    variants = experiment.get("model_variants") or {}
    for policy, variant in variants.items():
        candidate = _extract_candidate_variant(experiment, policy, variant)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _collect_qualifying_candidates(
    experiments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect qualifying candidates across the approved geometry experiments."""
    candidates: list[dict[str, Any]] = []
    for experiment in experiments:
        candidates.extend(_extract_experiment_candidates(experiment))
    return candidates


def _validate_candidate_collection(
    candidates: list[dict[str, Any]], expected_count: int
) -> None:
    """Require the expected number of unique candidates from the accepted geometry."""
    candidate_ids = {item["candidate_id"] for item in candidates}
    if len(candidates) != expected_count or len(candidate_ids) != expected_count:
        raise ValueError("v4h_qualifying_candidate_count_mismatch")


def extract_qualifying_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the strict Forward-Validation passes and their persisted scores."""
    validate_source_report(report)
    candidates = _collect_qualifying_candidates(report["experiments"])
    expected = int(report["acceptance_policy"]["strict_acceptance_count"])
    _validate_candidate_collection(candidates, expected)
    return candidates


def _same_utc_instant(left: Any, right: Any) -> bool:
    """Compare timezone-aware timestamps independent of ISO-8601 suffix style."""
    try:
        left_timestamp = pd.Timestamp(left)
        right_timestamp = pd.Timestamp(right)
    except (TypeError, ValueError):
        return False
    if left_timestamp.tzinfo is None or right_timestamp.tzinfo is None:
        return False
    return left_timestamp.tz_convert("UTC") == right_timestamp.tz_convert("UTC")


def validate_market_manifest(
    source_metadata: dict[str, Any],
    current_manifest: dict[str, Any],
    *,
    current_commit_sha: str,
) -> None:
    """Fail closed unless re-fetched candles exactly match the V4-H input hash."""
    expected_fields = (
        current_manifest.get("symbol") == source_metadata.get("symbol"),
        current_manifest.get("interval") == "1m",
        current_manifest.get("requested_days") == source_metadata.get("requested_days"),
        current_manifest.get("row_count") == source_metadata.get("row_count"),
        _same_utc_instant(
            current_manifest.get("start_utc"), source_metadata.get("start_utc")
        ),
        _same_utc_instant(
            current_manifest.get("end_utc"), source_metadata.get("end_utc")
        ),
        current_manifest.get("csv_sha256") == source_metadata.get("csv_sha256"),
        current_manifest.get("commit_sha") == current_commit_sha,
        current_manifest.get("public_api_only") is True,
        current_manifest.get("credentials_used") is False,
        current_manifest.get("missing_minute_intervals") == 0,
        current_manifest.get("private_api_calls") == 0,
        current_manifest.get("real_order_attempts") == 0,
    )
    if not all(expected_fields):
        raise ValueError("v4h_market_data_replay_integrity_mismatch")


def load_validated_minute_data(
    input_dir: Path,
    source_metadata: dict[str, Any],
    *,
    current_commit_sha: str,
) -> DataFrame:
    """Load minute candles only after provenance and byte-level digest checks."""
    symbol = str(source_metadata["symbol"])
    stem = f"binance-{symbol.lower()}-1m"
    csv_path = input_dir / f"{stem}.csv"
    manifest_path = input_dir / f"{stem}.manifest.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"economic_market_data_missing:{symbol}")
    manifest_bytes = manifest_path.read_bytes()
    current_manifest = json.loads(manifest_bytes.decode("utf-8"))
    csv_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if current_manifest.get("csv_sha256") != csv_digest:
        raise ValueError(f"economic_market_csv_digest_invalid:{symbol}")
    validate_market_manifest(
        source_metadata, current_manifest, current_commit_sha=current_commit_sha
    )
    candles = pd.read_csv(csv_path)
    expected_rows = int(source_metadata["row_count"])
    if len(candles) != expected_rows:
        raise ValueError(f"economic_market_row_count_invalid:{symbol}")
    candles["date"] = pd.to_datetime(candles["date"], utc=True, errors="coerce")
    if candles["date"].isna().any():
        raise ValueError(f"economic_market_timestamp_invalid:{symbol}")
    if (
        candles["date"].duplicated().any()
        or not candles["date"].is_monotonic_increasing
    ):
        raise ValueError(f"economic_market_timestamp_order_invalid:{symbol}")
    if len(candles) > 1 and not bool(
        (candles["date"].diff().iloc[1:] == pd.Timedelta(minutes=1)).all()
    ):
        raise ValueError(f"economic_market_gap_detected:{symbol}")
    return candles.loc[:, list(OHLCV_COLUMNS)]


def _validate_resample_input(candles: DataFrame, timeframe: str) -> int:
    """Validate resampling scope and required minute candle columns."""
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError("economic_timeframe_out_of_policy")
    if candles.empty or not set(OHLCV_COLUMNS).issubset(candles.columns):
        raise ValueError("economic_ohlcv_input_invalid")
    return TIMEFRAME_MINUTES[timeframe]


def _prepare_resample_source(candles: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Normalize and validate timestamp and numeric OHLCV columns."""
    source = candles.loc[:, list(OHLCV_COLUMNS)].copy().reset_index(drop=True)
    source["date"] = pd.to_datetime(source["date"], utc=True, errors="coerce")
    numeric = source.loc[:, ["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if source["date"].isna().any():
        raise ValueError("economic_ohlcv_values_invalid")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("economic_ohlcv_values_invalid")
    if source["date"].duplicated().any() or not source["date"].is_monotonic_increasing:
        raise ValueError("economic_ohlcv_timestamps_invalid")
    return source, numeric


def _ohlcv_geometry_is_valid(numeric: DataFrame) -> bool:
    """Check price positivity, volume, and candle high/low consistency."""
    prices = ("open", "high", "low", "close")
    if any((numeric[column] <= 0).any() for column in prices):
        return False
    if (numeric["volume"] < 0).any():
        return False
    high = numeric["high"]
    low = numeric["low"]
    upper_bound = numeric[["open", "close"]].max(axis=1)
    lower_bound = numeric[["open", "close"]].min(axis=1)
    return bool(
        (high >= upper_bound).all()
        and (low <= lower_bound).all()
        and (high >= low).all()
    )


def _aggregate_complete_bars(source: DataFrame, minutes: int) -> DataFrame:
    """Apply the V4-H epoch-aligned OHLCV aggregation rule."""
    grouped = source.set_index("date").resample(
        f"{minutes}min", origin="epoch", closed="left", label="left"
    )
    bars = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    complete = grouped["close"].count() == minutes
    return bars.loc[complete].dropna().reset_index()


def resample_ohlcv(candles: DataFrame, timeframe: str) -> DataFrame:
    """Rebuild complete epoch-aligned OHLCV bars using the V4-H aggregation rule."""
    minutes = _validate_resample_input(candles, timeframe)
    source, numeric = _prepare_resample_source(candles)
    if not _ohlcv_geometry_is_valid(numeric):
        raise ValueError("economic_ohlcv_geometry_invalid")
    source.loc[:, ["open", "high", "low", "close", "volume"]] = numeric
    result = _aggregate_complete_bars(source, minutes)
    if result.empty:
        raise ValueError("economic_resampled_bars_empty")
    return result.loc[:, list(OHLCV_COLUMNS)]


def _validate_hold_threshold(hold_threshold: float) -> float:
    """Validate the persisted V4-H threshold before signal conversion."""
    threshold = float(hold_threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("economic_hold_threshold_invalid")
    return threshold


def _validated_probability_values(
    probabilities: dict[str, Any],
) -> tuple[float, float, float]:
    """Validate and return BUY, SELL, and HOLD probabilities."""
    try:
        buy = float(probabilities["BUY"])
        sell = float(probabilities["SELL"])
        hold = float(probabilities["HOLD"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("economic_probabilities_invalid") from None
    values = np.asarray([buy, sell, hold], dtype=float)
    if not bool(np.isfinite(values).all()):
        raise ValueError("economic_probabilities_invalid")
    if bool(np.any(values < 0)) or bool(np.any(values > 1)):
        raise ValueError("economic_probabilities_invalid")
    if abs(float(values.sum()) - 1.0) > 1e-6:
        raise ValueError("economic_probabilities_invalid")
    return buy, sell, hold


def probability_to_signal(probabilities: dict[str, Any], hold_threshold: float) -> str:
    """Apply the same HOLD-threshold and BUY/SELL tie-break rule as V4-H."""
    threshold = _validate_hold_threshold(hold_threshold)
    buy, sell, hold = _validated_probability_values(probabilities)
    if hold >= threshold and hold > max(buy, sell):
        return "HOLD"
    return "BUY" if buy >= sell else "SELL"


def _validate_confidence_threshold(confidence_threshold: float) -> float:
    """Validate an optional directional entry-confidence floor."""
    threshold = float(confidence_threshold)
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("economic_confidence_threshold_invalid")
    return threshold


def _forward_signal_record(
    row: dict[str, Any],
    hold_threshold: float,
    confidence_floor: float | None,
    delta: timedelta,
) -> dict[str, Any]:
    """Transform and confidence-filter one persisted probability record."""
    row_index = int(row["row_index"])
    timestamp = pd.Timestamp(row["timestamp"])
    if timestamp.tzinfo is None:
        raise ValueError("economic_forward_signal_order_invalid")
    timestamp = timestamp.tz_convert("UTC")
    probabilities = row["probabilities"]
    buy, sell, hold = _validated_probability_values(probabilities)
    raw_signal = probability_to_signal(probabilities, hold_threshold)
    confidence_passed = (
        raw_signal != "BUY" or confidence_floor is None or buy >= confidence_floor
    )
    signal = raw_signal if confidence_passed else NO_ENTRY_SIGNAL
    directional_confidence = (
        buy if raw_signal == "BUY" else sell if raw_signal == "SELL" else None
    )
    return {
        "row_index": row_index,
        "signal_timestamp": timestamp,
        "execution_timestamp": timestamp + delta,
        "signal": signal,
        "raw_signal": raw_signal,
        "buy_probability": buy,
        "sell_probability": sell,
        "hold_probability": hold,
        "directional_confidence": directional_confidence,
        "confidence_threshold": confidence_floor,
        "confidence_passed": confidence_passed,
    }


def build_forward_signals(
    candidate: dict[str, Any],
    confidence_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Transform persisted probabilities and optionally filter BUY entries.

    The default None preserves the historical V4-J signal transformation.
    V4-K passes an explicit confidence floor. SELL exits are never filtered.
    """
    rows = candidate["probability_records"]
    hold_threshold = float(candidate["selected_hold_threshold"])
    confidence_floor = (
        None
        if confidence_threshold is None
        else _validate_confidence_threshold(confidence_threshold)
    )
    delta = timedelta(minutes=TIMEFRAME_MINUTES[candidate["timeframe"]])
    signals: list[dict[str, Any]] = []
    previous_index = -1
    previous_timestamp: pd.Timestamp | None = None
    for row in rows:
        item = _forward_signal_record(row, hold_threshold, confidence_floor, delta)
        timestamp = pd.Timestamp(item["signal_timestamp"])
        row_index = int(item["row_index"])
        if row_index <= previous_index:
            raise ValueError("economic_forward_signal_order_invalid")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("economic_forward_signal_timestamp_invalid")
        signals.append(item)
        previous_index = row_index
        previous_timestamp = timestamp
    if len(signals) != int(candidate["forward_row_count"]):
        raise ValueError("economic_forward_signal_count_mismatch")
    return signals


def _open_position(
    equity: float,
    direction: int,
    reference_price: float,
    timestamp: pd.Timestamp,
    atr_value: float,
    entry_bar_index: int,
    entry_confidence: float,
    atr_reference_timestamp: pd.Timestamp,
    config: ExecutionConfig,
    management: TradeManagementConfig,
) -> tuple[OpenPosition, float]:
    """Open a 1x-equity Spot long and attach causal ATR barriers."""
    fee_rate = config.fee_bps / 10_000.0
    slippage_rate = config.slippage_bps / 10_000.0
    quantity = equity * config.position_fraction / reference_price
    entry_fill = reference_price * (1.0 + direction * slippage_rate)
    entry_fee = quantity * entry_fill * fee_rate
    entry_slippage = quantity * abs(entry_fill - reference_price)
    take_profit = entry_fill + management.take_profit_atr_multiplier * atr_value
    stop_loss = entry_fill - management.stop_loss_atr_multiplier * atr_value
    if not np.isfinite(take_profit) or not np.isfinite(stop_loss) or stop_loss <= 0:
        raise ValueError("economic_atr_barrier_price_invalid")
    position = OpenPosition(
        direction=direction,
        entry_timestamp=timestamp.isoformat(),
        entry_reference_price=reference_price,
        entry_fill_price=entry_fill,
        quantity=quantity,
        equity_at_entry=equity,
        entry_fee=entry_fee,
        entry_slippage_cost=entry_slippage,
        atr_value=atr_value,
        take_profit_price=take_profit,
        stop_loss_price=stop_loss,
        entry_bar_index=entry_bar_index,
        entry_confidence=entry_confidence,
        atr_reference_timestamp=atr_reference_timestamp.isoformat(),
        initial_stop_loss_price=stop_loss,
        high_water_mark=entry_fill,
        trailing_stop_active=False,
        trailing_stop_steps=0,
        time_exit_candles=management.time_exit_candles,
        time_exit_mfe_threshold_atr=management.time_exit_mfe_threshold_atr,
        time_exit_evaluated=False,
        time_exit_pending=False,
        max_favorable_excursion_atr=0.0,
        max_adverse_excursion_atr=0.0,
    )
    marked_equity = equity - entry_fee - entry_slippage
    if marked_equity <= 0:
        raise ValueError("economic_equity_non_positive_after_entry")
    return position, marked_equity


def _mark_position(position: OpenPosition, mark_price: float) -> float:
    """Mark a position to market after its entry fee and slippage."""
    market_pnl = (
        position.quantity
        * position.direction
        * (mark_price - position.entry_reference_price)
    )
    return (
        position.equity_at_entry
        + market_pnl
        - position.entry_fee
        - position.entry_slippage_cost
    )


def _update_position_excursions(state: _SimulationState, bar: Any) -> None:
    """Update cumulative OHLC-based MFE and MAE in units of entry ATR."""
    position = state.position
    if position is None:
        return
    high, low = float(bar["high"]), float(bar["low"])
    if position.direction == 1:
        favorable = max(0.0, (high - position.entry_fill_price) / position.atr_value)
        adverse = max(0.0, (position.entry_fill_price - low) / position.atr_value)
    else:
        favorable = max(0.0, (position.entry_fill_price - low) / position.atr_value)
        adverse = max(0.0, (high - position.entry_fill_price) / position.atr_value)
    state.position = replace(
        position,
        max_favorable_excursion_atr=max(
            position.max_favorable_excursion_atr, favorable
        ),
        max_adverse_excursion_atr=max(position.max_adverse_excursion_atr, adverse),
    )


def _stop_exit_type(position: OpenPosition) -> str:
    """Classify the protective stop that was active when an exit occurred."""
    if not position.trailing_stop_active:
        return "INITIAL_STOP"
    if position.trailing_stop_steps == 0:
        return "BREAKEVEN_STOP"
    return "TRAILING_STOP"


def _stop_exit_reason(position: OpenPosition, collision: bool) -> str:
    """Preserve stop-first collisions while identifying trailing-stop exits."""
    if collision:
        return "COLLISION"
    return {
        "INITIAL_STOP": "SL_HIT",
        "BREAKEVEN_STOP": "BREAKEVEN_STOP_HIT",
        "TRAILING_STOP": "TRAILING_STOP_HIT",
    }[_stop_exit_type(position)]


def _net_breakeven_reference_price(
    position: OpenPosition, config: ExecutionConfig
) -> float:
    """Solve the exit reference price that offsets both-side fees and slippage."""
    fee_rate = config.fee_bps / 10_000.0
    slippage_rate = config.slippage_bps / 10_000.0
    denominator = 1.0 - (fee_rate * (1.0 - slippage_rate) + slippage_rate)
    if denominator <= 0 or position.quantity <= 0:
        raise ValueError("economic_trailing_breakeven_unreachable")
    entry_cost_per_unit = (
        position.entry_fee + position.entry_slippage_cost
    ) / position.quantity
    reference_price = (
        position.entry_reference_price + entry_cost_per_unit
    ) / denominator
    if not np.isfinite(reference_price) or reference_price <= 0:
        raise ValueError("economic_trailing_breakeven_unreachable")
    return reference_price


def _close_position(
    position: OpenPosition,
    reference_price: float,
    timestamp: pd.Timestamp,
    reason: str,
    config: ExecutionConfig,
    *,
    exit_bar_index: int,
    exit_trigger_type: str,
) -> tuple[dict[str, Any], float]:
    """Close one Spot long with exit-side fee and adverse slippage."""
    fee_rate = config.fee_bps / 10_000.0
    slippage_rate = config.slippage_bps / 10_000.0
    exit_fill = reference_price * (1.0 - position.direction * slippage_rate)
    exit_fee = position.quantity * exit_fill * fee_rate
    exit_slippage = position.quantity * abs(exit_fill - reference_price)
    gross_pnl = (
        position.quantity
        * position.direction
        * (reference_price - position.entry_reference_price)
    )
    fees = position.entry_fee + exit_fee
    slippage = position.entry_slippage_cost + exit_slippage
    net_pnl = gross_pnl - fees - slippage
    equity = position.equity_at_entry + net_pnl
    if equity <= 0:
        raise ValueError("economic_equity_non_positive_after_exit")
    bars_held = exit_bar_index - position.entry_bar_index + 1
    if bars_held < 1:
        raise RuntimeError("economic_position_holding_index_invalid")
    stop_exit_type = (
        _stop_exit_type(position)
        if reason in {"SL_HIT", "BREAKEVEN_STOP_HIT", "TRAILING_STOP_HIT", "COLLISION"}
        else None
    )
    breakeven_stop_price = (
        _net_breakeven_reference_price(position, config)
        if position.trailing_stop_active
        else None
    )
    trade = {
        "direction": "LONG" if position.direction == 1 else "SHORT",
        "entry_timestamp": position.entry_timestamp,
        "exit_timestamp": timestamp.isoformat(),
        "entry_reference_price": round(position.entry_reference_price, 10),
        "entry_fill_price": round(position.entry_fill_price, 10),
        "entry_atr_reference_timestamp": position.atr_reference_timestamp,
        "atr_value": round(position.atr_value, 10),
        "barrier_reference_price": round(position.entry_fill_price, 10),
        "initial_stop_loss_price": round(position.initial_stop_loss_price, 10),
        "stop_loss_price": round(position.stop_loss_price, 10),
        "take_profit_price": round(position.take_profit_price, 10),
        "high_water_mark": round(position.high_water_mark, 10),
        "trailing_stop_active": position.trailing_stop_active,
        "trailing_stop_steps": position.trailing_stop_steps,
        "breakeven_stop_price": (
            round(breakeven_stop_price, 10)
            if breakeven_stop_price is not None
            else None
        ),
        "stop_exit_type": stop_exit_type,
        "entry_confidence": round(position.entry_confidence, 10),
        "bars_held": bars_held,
        "mfe_atr": round(position.max_favorable_excursion_atr, 8),
        "mae_atr": round(position.max_adverse_excursion_atr, 8),
        "time_exit_candles": position.time_exit_candles,
        "time_exit_mfe_threshold_atr": round(position.time_exit_mfe_threshold_atr, 8),
        "time_exit_evaluated": position.time_exit_evaluated,
        "time_exit_pending": position.time_exit_pending,
        "time_exit_triggered": reason == "TIME_EXIT",
        "exit_reference_price": round(reference_price, 10),
        "exit_fill_price": round(exit_fill, 10),
        "exit_trigger_type": exit_trigger_type,
        "quantity": round(position.quantity, 10),
        "gross_market_pnl_usd": round(gross_pnl, 8),
        "fees_usd": round(fees, 8),
        "slippage_cost_usd": round(slippage, 8),
        "net_pnl_usd": round(net_pnl, 8),
        "net_return_fraction": round(net_pnl / position.equity_at_entry, 10),
        "exit_reason": reason,
    }
    return trade, equity


def max_drawdown_from_equity(equity_values: list[float]) -> float:
    """Return peak-to-trough drawdown as a positive fraction."""
    if not equity_values or not all(
        np.isfinite(float(value)) for value in equity_values
    ):
        raise ValueError("economic_equity_curve_invalid")
    peak = float(equity_values[0])
    maximum = 0.0
    for raw in equity_values:
        equity = float(raw)
        if equity <= 0:
            raise ValueError("economic_equity_non_positive")
        peak = max(peak, equity)
        maximum = max(maximum, (peak - equity) / peak)
    return maximum


def _trade_pnl_groups(
    trades: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    """Separate positive and negative net trade outcomes."""
    pnls = [float(item["net_pnl_usd"]) for item in trades]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    return wins, losses


def _profit_factor_details(
    wins: list[float], losses: list[float]
) -> tuple[float | None, bool, float, float]:
    """Return net profit factor, no-loss status, and gross outcome totals."""
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    infinite = bool(wins and not losses)
    return profit_factor, infinite, gross_profit, gross_loss


def _optional_mean(values: list[float]) -> float | None:
    """Return a rounded mean for a non-empty trade outcome group."""
    return round(float(np.mean(values)), 8) if values else None


def _excursion_statistics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize MFE and MAE for all trades and negative-PnL trades."""
    mfe_values = [float(item["mfe_atr"]) for item in trades]
    mae_values = [float(item["mae_atr"]) for item in trades]
    losing = [item for item in trades if float(item["net_pnl_usd"]) < 0]
    losing_mfe = [float(item["mfe_atr"]) for item in losing]
    losing_mae = [float(item["mae_atr"]) for item in losing]
    return {
        "mean_mfe_atr": _optional_mean(mfe_values),
        "mean_mae_atr": _optional_mean(mae_values),
        "mean_losing_trade_mfe_atr": _optional_mean(losing_mfe),
        "mean_losing_trade_mae_atr": _optional_mean(losing_mae),
        "losing_trades_with_positive_mfe_count": sum(value > 0 for value in losing_mfe),
        "losing_trades_with_mfe_at_least_0_5_atr_count": sum(
            value >= 0.5 for value in losing_mfe
        ),
    }


def _metrics(
    trades: list[dict[str, Any]],
    equity_values: list[float],
    config: ExecutionConfig,
) -> dict[str, Any]:
    """Calculate net-of-cost portfolio and trade statistics."""
    wins, losses = _trade_pnl_groups(trades)
    pnls = [float(item["net_pnl_usd"]) for item in trades]
    pf, pf_infinite, gross_profit, gross_loss = _profit_factor_details(wins, losses)
    final_equity = float(equity_values[-1])
    max_drawdown = max_drawdown_from_equity(equity_values)
    trade_count = len(trades)
    pf_pass = pf is not None and pf > 1.5
    drawdown_pass = max_drawdown < config.maximum_drawdown
    excursion_statistics = _excursion_statistics(trades)
    return {
        "initial_equity_usd": round(config.initial_equity, 8),
        "final_equity_usd": round(final_equity, 8),
        "net_return_fraction": round(final_equity / config.initial_equity - 1.0, 10),
        "net_return_percent": round(
            (final_equity / config.initial_equity - 1.0) * 100.0, 8
        ),
        "trade_count": trade_count,
        "win_rate": round(len(wins) / trade_count, 8) if trade_count else 0.0,
        "profit_factor_net_of_costs": round(pf, 8) if pf is not None else None,
        "profit_factor_infinite_no_losses": pf_infinite,
        "gross_profit_usd": round(gross_profit, 8),
        "gross_loss_usd": round(gross_loss, 8),
        "average_win_usd": _optional_mean(wins),
        "average_loss_usd": _optional_mean(losses),
        "average_net_pnl_per_trade_usd": _optional_mean(pnls),
        "max_drawdown_fraction": round(max_drawdown, 10),
        "max_drawdown_percent": round(max_drawdown * 100.0, 8),
        "profit_factor_gate_passed": pf_pass,
        "drawdown_gate_passed": drawdown_pass,
        "economic_candidate_gate_passed": bool(pf_pass and drawdown_pass),
        "maximum_drawdown_limit_fraction": config.maximum_drawdown,
        **excursion_statistics,
    }


def _validate_drawdown_accounting(
    state_drawdown: float, equity_values: list[float]
) -> None:
    """Compare full-precision state drawdown with the rounded equity-curve series."""
    curve_drawdown = max_drawdown_from_equity(equity_values)
    if state_drawdown > curve_drawdown + MAX_DRAWDOWN_ACCOUNTING_TOLERANCE:
        raise RuntimeError(
            "economic_drawdown_accounting_inconsistent:"
            f"state={state_drawdown:.12g};curve={curve_drawdown:.12g}"
        )


def _validate_execution_timestamps(bars: DataFrame) -> None:
    """Require unique, monotonic UTC candle timestamps."""
    dates = bars["date"]
    if (
        dates.isna().any()
        or dates.duplicated().any()
        or not dates.is_monotonic_increasing
    ):
        raise ValueError("economic_execution_timestamps_invalid")


def _validate_execution_column(bars: DataFrame, column: str) -> None:
    """Require finite OHLCV values within the appropriate sign domain."""
    values = bars[column].to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()):
        raise ValueError("economic_execution_ohlcv_non_finite")
    if column == "volume":
        if bool(np.any(values < 0)):
            raise ValueError("economic_execution_volume_invalid")
        return
    if bool(np.any(values <= 0)):
        raise ValueError("economic_execution_prices_invalid")


def _validate_execution_geometry(bars: DataFrame) -> None:
    """Require high/low to bound each candle's open and close."""
    invalid_geometry = (
        bool((bars["high"] < bars["low"]).any())
        or bool((bars["high"] < bars[["open", "close"]].max(axis=1)).any())
        or bool((bars["low"] > bars[["open", "close"]].min(axis=1)).any())
    )
    if invalid_geometry:
        raise ValueError("economic_execution_ohlcv_geometry_invalid")


def _normalize_execution_bars(bars: DataFrame) -> DataFrame:
    """Validate timestamped OHLCV data required for causal ATR and execution."""
    if bars.empty or not set(OHLCV_COLUMNS).issubset(bars.columns):
        raise ValueError("economic_execution_bars_invalid")
    normalized = bars.loc[:, list(OHLCV_COLUMNS)].copy().reset_index(drop=True)
    normalized["date"] = pd.to_datetime(normalized["date"], utc=True, errors="coerce")
    _validate_execution_timestamps(normalized)
    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        _validate_execution_column(normalized, column)
    _validate_execution_geometry(normalized)
    return normalized


def _signal_market_indices(
    item: dict[str, Any],
    index_by_time: dict[pd.Timestamp, int],
) -> tuple[int, int]:
    """Resolve signal and execution indices, enforcing next-bar causality."""
    execution_time = pd.Timestamp(item["execution_timestamp"])
    if execution_time.tzinfo is None:
        raise ValueError("economic_execution_signal_timezone_required")
    execution_index = index_by_time.get(execution_time.tz_convert("UTC"))
    if execution_index is None:
        raise ValueError("economic_signal_price_timestamp_mismatch")
    signal_time = item.get("signal_timestamp")
    if signal_time is None:
        signal_index = execution_index - 1
    else:
        signal_timestamp = pd.Timestamp(signal_time)
        if signal_timestamp.tzinfo is None:
            raise ValueError("economic_signal_timestamp_timezone_required")
        signal_index_value = index_by_time.get(signal_timestamp.tz_convert("UTC"))
        if signal_index_value is None:
            raise ValueError("economic_signal_price_timestamp_mismatch")
        signal_index = signal_index_value
    if signal_index < 0 or signal_index + 1 != execution_index:
        raise ValueError("economic_signal_next_bar_alignment_invalid")
    return signal_index, execution_index


def _index_execution_signals(
    signals: list[dict[str, Any]],
    index_by_time: dict[pd.Timestamp, int],
) -> dict[int, dict[str, Any]]:
    """Map persisted signal and execution times to adjacent exact market bars."""
    action_by_index: dict[int, dict[str, Any]] = {}
    for raw_item in signals:
        item = dict(raw_item)
        signal_index, execution_index = _signal_market_indices(item, index_by_time)
        signal = item.get("signal")
        if signal not in (*LABELS, NO_ENTRY_SIGNAL):
            raise ValueError("economic_signal_contract_invalid")
        if execution_index in action_by_index:
            raise ValueError("economic_signal_contract_invalid")
        item["signal_index"] = signal_index
        action_by_index[execution_index] = item
    if not action_by_index:
        raise ValueError("economic_bars_or_signals_empty")
    return action_by_index


@dataclass
class _SimulationState:
    """Mutable accounting state for one isolated candidate replay."""

    position: OpenPosition | None
    cash_equity: float
    peak: float
    maximum_drawdown: float
    trades: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]


def _append_equity_event(
    state: _SimulationState,
    timestamp: pd.Timestamp,
    equity: float,
    event: str,
) -> None:
    """Append one timestamped account-equity event."""
    state.equity_curve.append(
        {
            "timestamp": timestamp.isoformat(),
            "equity_usd": round(equity, 8),
            "event": event,
        }
    )


def _update_equity_extrema(state: _SimulationState, equity: float) -> None:
    """Update peak and drawdown at every realized or marked equity event."""
    if not np.isfinite(equity) or equity <= 0:
        raise ValueError("economic_equity_non_positive")
    state.peak = max(state.peak, equity)
    state.maximum_drawdown = max(
        state.maximum_drawdown, (state.peak - equity) / state.peak
    )


def _close_state_position(
    state: _SimulationState,
    reference_price: float,
    timestamp: pd.Timestamp,
    bar_index: int,
    reason: str,
    trigger_type: str,
    config: ExecutionConfig,
) -> bool:
    """Close the active Spot long once and record realized equity immediately."""
    position = state.position
    if position is None:
        return False
    trade, equity = _close_position(
        position,
        reference_price,
        timestamp,
        reason,
        config,
        exit_bar_index=bar_index,
        exit_trigger_type=trigger_type,
    )
    state.trades.append(trade)
    state.position = None
    state.cash_equity = equity
    _update_equity_extrema(state, equity)
    _append_equity_event(state, timestamp, equity, f"POSITION_CLOSED_{trigger_type}")
    return True


def _close_for_signal(
    state: _SimulationState,
    signal: str,
    bar: Any,
    timestamp: pd.Timestamp,
    bar_index: int,
    config: ExecutionConfig,
) -> bool:
    """Treat SELL as a risk-reducing close; HOLD and NO_ENTRY keep the long."""
    if signal != "SELL":
        return False
    return _close_state_position(
        state, float(bar["open"]), timestamp, bar_index, "SELL", "MODEL_SELL", config
    )


def _open_for_signal(
    state: _SimulationState,
    signal_item: dict[str, Any],
    bar: Any,
    timestamp: pd.Timestamp,
    bar_index: int,
    atr_values: pd.Series,
    config: ExecutionConfig,
    management: TradeManagementConfig,
    normalized: DataFrame,
) -> None:
    """Open one confident BUY only when flat, using signal-time ATR."""
    if signal_item["signal"] != "BUY" or state.position is not None:
        return
    signal_index = int(signal_item["signal_index"])
    atr_value = float(atr_values.iloc[signal_index])
    if not np.isfinite(atr_value) or atr_value <= 0:
        raise ValueError("economic_atr_unavailable_for_entry")
    confidence = float(signal_item.get("buy_probability", 1.0))
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("economic_entry_confidence_invalid")
    atr_timestamp = pd.Timestamp(normalized["date"].iloc[signal_index])
    state.position, marked_equity = _open_position(
        state.cash_equity,
        1,
        float(bar["open"]),
        timestamp,
        atr_value,
        bar_index,
        confidence,
        atr_timestamp,
        config,
        management,
    )
    _update_equity_extrema(state, marked_equity)
    _append_equity_event(state, timestamp, marked_equity, "POSITION_OPENED")


def _gap_barrier_touch(
    position: OpenPosition, bar: Any
) -> tuple[str, float, str] | None:
    """Detect an opening gap through the initial or currently ratcheted stop."""
    opened = float(bar["open"])
    if opened <= position.stop_loss_price:
        return _stop_exit_reason(position, False), opened, "OPEN_GAP"
    if opened >= position.take_profit_price:
        return "TP_HIT", opened, "OPEN_GAP"
    return None


def _intrabar_barrier_touch(
    position: OpenPosition,
    bar: Any,
) -> tuple[str, float, str] | None:
    """Resolve OHLC touches with stop-first collision handling."""
    stop_hit = float(bar["low"]) <= position.stop_loss_price
    take_hit = float(bar["high"]) >= position.take_profit_price
    if stop_hit:
        reason = _stop_exit_reason(position, take_hit)
        trigger = "INTRABAR_COLLISION" if take_hit else "INTRABAR_STOP"
        return reason, position.stop_loss_price, trigger
    if take_hit:
        return "TP_HIT", position.take_profit_price, "INTRABAR_TAKE"
    return None


def _close_on_barrier(
    state: _SimulationState,
    bar: Any,
    timestamp: pd.Timestamp,
    bar_index: int,
    config: ExecutionConfig,
    *,
    gap_first: bool,
) -> bool:
    """Check an opening gap or intrabar barrier with stop-first collision logic."""
    if state.position is None:
        return False
    touch = (
        _gap_barrier_touch(state.position, bar)
        if gap_first
        else _intrabar_barrier_touch(state.position, bar)
    )
    if touch is None:
        return False
    reason, reference_price, trigger = touch
    return _close_state_position(
        state, reference_price, timestamp, bar_index, reason, trigger, config
    )


def _update_trailing_stop_after_completed_bar(
    state: _SimulationState,
    bar: Any,
    config: ExecutionConfig,
    management: TradeManagementConfig,
) -> None:
    """Ratchet the stop after a completed bar; the new stop applies next bar."""
    position = state.position
    if position is None or not management.trailing_stop_enabled:
        return
    high_water_mark = max(position.high_water_mark, float(bar["high"]))
    activation_price = (
        position.entry_fill_price
        + management.trailing_activation_atr * position.atr_value
    )
    if high_water_mark < activation_price:
        state.position = replace(position, high_water_mark=high_water_mark)
        return
    step_size = management.trailing_step_atr * position.atr_value
    steps = int(np.floor((high_water_mark - activation_price) / step_size + 1e-12))
    steps = max(position.trailing_stop_steps, steps)
    breakeven = _net_breakeven_reference_price(position, config)
    proposed_stop = breakeven + steps * step_size
    state.position = replace(
        position,
        high_water_mark=high_water_mark,
        trailing_stop_active=True,
        trailing_stop_steps=steps,
        stop_loss_price=max(position.stop_loss_price, proposed_stop),
    )


def _close_on_pending_time_exit(
    state: _SimulationState,
    bar: Any,
    timestamp: pd.Timestamp,
    bar_index: int,
    config: ExecutionConfig,
) -> bool:
    """Execute a previously scheduled exit at the next candle open."""
    position = state.position
    if position is None or not position.time_exit_pending:
        return False
    return _close_state_position(
        state,
        float(bar["open"]),
        timestamp,
        bar_index,
        "TIME_EXIT",
        "TIME_EXIT_OPEN",
        config,
    )


def _schedule_time_exit_after_completed_bar(
    state: _SimulationState,
    bar_index: int,
) -> None:
    """Evaluate the MFE threshold once X complete candles have elapsed."""
    position = state.position
    if (
        position is None
        or position.time_exit_candles is None
        or position.time_exit_evaluated
    ):
        return
    bars_held = bar_index - position.entry_bar_index + 1
    if bars_held < position.time_exit_candles:
        return
    state.position = replace(
        position,
        time_exit_evaluated=True,
        time_exit_pending=(
            position.max_favorable_excursion_atr < position.time_exit_mfe_threshold_atr
        ),
    )


def _close_on_vertical_barrier(
    state: _SimulationState,
    bar: Any,
    timestamp: pd.Timestamp,
    bar_index: int,
    management: TradeManagementConfig,
    config: ExecutionConfig,
) -> bool:
    """Close at candle close after the configured Triple-Barrier horizon."""
    if state.position is None:
        return False
    bars_held = bar_index - state.position.entry_bar_index + 1
    if bars_held < management.horizon_candles:
        return False
    return _close_state_position(
        state,
        float(bar["close"]),
        timestamp,
        bar_index,
        "HORIZON_EXPIRY",
        "VERTICAL_BARRIER",
        config,
    )


def _mark_execution_bar_close(
    state: _SimulationState, bar: Any, timestamp: pd.Timestamp
) -> None:
    """Mark active exposure to the bar close and update peak-to-trough drawdown."""
    if state.position is None:
        marked_equity = state.cash_equity
    else:
        marked_equity = _mark_position(state.position, float(bar["close"]))
    _update_equity_extrema(state, marked_equity)
    _append_equity_event(state, timestamp, marked_equity, "BAR_CLOSE_MARK")


def _force_close_forward_window(
    state: _SimulationState,
    final_bar: Any,
    timestamp: pd.Timestamp,
    bar_index: int,
    config: ExecutionConfig,
) -> None:
    """Close a remaining position at the Forward-window boundary."""
    _close_state_position(
        state,
        float(final_bar["close"]),
        timestamp,
        bar_index,
        "END_OF_FORWARD_WINDOW",
        "FORWARD_WINDOW",
        config,
    )


def _execution_atr_values(
    normalized: DataFrame,
    atr_values: pd.Series | None,
    management: TradeManagementConfig,
) -> pd.Series:
    """Use supplied deterministic test ATR or causal ATR from normalized bars."""
    if atr_values is None:
        return calculate_atr(normalized, period=management.atr_period)
    if len(atr_values) != len(normalized):
        raise ValueError("economic_atr_length_mismatch")
    return pd.to_numeric(atr_values, errors="coerce").reset_index(drop=True)


def _new_simulation_state(
    normalized: DataFrame,
    action_by_index: dict[int, dict[str, Any]],
    config: ExecutionConfig,
) -> tuple[_SimulationState, int, int]:
    """Initialize the account ledger and exact Forward execution interval."""
    first_index = min(action_by_index)
    last_index = max(action_by_index)
    first_timestamp = pd.Timestamp(normalized["date"].iloc[first_index])
    state = _SimulationState(
        position=None,
        cash_equity=float(config.initial_equity),
        peak=float(config.initial_equity),
        maximum_drawdown=0.0,
        trades=[],
        equity_curve=[
            {
                "timestamp": first_timestamp.isoformat(),
                "equity_usd": round(float(config.initial_equity), 8),
                "event": "START",
            }
        ],
    )
    return state, first_index, last_index


def _process_execution_bar(
    state: _SimulationState,
    index: int,
    normalized: DataFrame,
    action_by_index: dict[int, dict[str, Any]],
    selected_atr: pd.Series,
    config: ExecutionConfig,
    management: TradeManagementConfig,
) -> None:
    """Process one candle in causal opening, signal, barrier, and mark order."""
    bar = normalized.iloc[index]
    timestamp = pd.Timestamp(bar["date"])
    gap_closed = _close_on_barrier(state, bar, timestamp, index, config, gap_first=True)
    time_exit_closed = False
    if not gap_closed:
        time_exit_closed = _close_on_pending_time_exit(
            state, bar, timestamp, index, config
        )
    signal_item = action_by_index.get(index)
    if signal_item is not None and not gap_closed and not time_exit_closed:
        signal = str(signal_item["signal"])
        _close_for_signal(state, signal, bar, timestamp, index, config)
        _open_for_signal(
            state,
            signal_item,
            bar,
            timestamp,
            index,
            selected_atr,
            config,
            management,
            normalized,
        )
    _update_position_excursions(state, bar)
    _close_on_barrier(state, bar, timestamp, index, config, gap_first=False)
    _update_trailing_stop_after_completed_bar(state, bar, config, management)
    _schedule_time_exit_after_completed_bar(state, index)
    _close_on_vertical_barrier(state, bar, timestamp, index, management, config)
    _mark_execution_bar_close(state, bar, timestamp)


def _validate_spot_simulation_result(state: _SimulationState) -> list[float]:
    """Reject forbidden directions and reconcile peak-to-trough accounting."""
    if any(item["direction"] != "LONG" for item in state.trades):
        raise RuntimeError("economic_spot_only_position_invariant_violation")
    equity_values = [float(item["equity_usd"]) for item in state.equity_curve]
    _validate_drawdown_accounting(state.maximum_drawdown, equity_values)
    return equity_values


def _simulation_exit_summary(
    trades: list[dict[str, Any]],
    signals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build exit-reason and confidence-filter counters for a candidate replay."""
    exit_reasons = sorted({str(item["exit_reason"]) for item in trades})
    stop_types = ("INITIAL_STOP", "BREAKEVEN_STOP", "TRAILING_STOP")
    stop_exit_type_counts = {
        stop_type: sum(item.get("stop_exit_type") == stop_type for item in trades)
        for stop_type in stop_types
    }
    return {
        "exit_reason_counts": {
            reason: sum(item["exit_reason"] == reason for item in trades)
            for reason in exit_reasons
        },
        "stop_exit_type_counts": stop_exit_type_counts,
        "initial_stop_exit_count": stop_exit_type_counts["INITIAL_STOP"],
        "breakeven_stop_exit_count": stop_exit_type_counts["BREAKEVEN_STOP"],
        "trailing_stop_exit_count": stop_exit_type_counts["TRAILING_STOP"],
        "time_exit_count": sum(item["exit_reason"] == "TIME_EXIT" for item in trades),
        "confidence_filter": {
            "threshold": None,
            "raw_buy_signal_count": sum(
                item.get("raw_signal", item["signal"]) == "BUY" for item in signals
            ),
            "accepted_buy_signal_count": sum(
                item["signal"] == "BUY" for item in signals
            ),
            "rejected_buy_signal_count": sum(
                item["signal"] == NO_ENTRY_SIGNAL for item in signals
            ),
            "sell_exit_signals_unfiltered": True,
        },
    }


def simulate_signals(
    bars: DataFrame,
    signals: list[dict[str, Any]],
    config: ExecutionConfig,
    *,
    trade_management_config: TradeManagementConfig | None = None,
    atr_values: pd.Series | None = None,
) -> dict[str, Any]:
    """Replay Spot entries with ATR barriers, SELL exits, and vertical expiry."""
    validate_execution_config(config)
    management = trade_management_config or TradeManagementConfig()
    validate_trade_management_config(management)
    if not signals:
        raise ValueError("economic_bars_or_signals_empty")
    normalized = _normalize_execution_bars(bars)
    selected_atr = _execution_atr_values(normalized, atr_values, management)
    index_by_time = {
        timestamp: index for index, timestamp in enumerate(normalized["date"].tolist())
    }
    action_by_index = _index_execution_signals(signals, index_by_time)
    state, first_index, last_index = _new_simulation_state(
        normalized, action_by_index, config
    )
    for index in range(first_index, last_index + 1):
        _process_execution_bar(
            state, index, normalized, action_by_index, selected_atr, config, management
        )
    final_bar = normalized.iloc[last_index]
    _force_close_forward_window(
        state, final_bar, pd.Timestamp(final_bar["date"]), last_index, config
    )
    equity_values = _validate_spot_simulation_result(state)
    metrics = _metrics(state.trades, equity_values, config)
    summary = _simulation_exit_summary(state.trades, signals)
    summary["confidence_filter"]["threshold"] = management.confidence_threshold
    return {
        "metrics": metrics,
        "trades": state.trades,
        "equity_curve": state.equity_curve,
        "total_fees_usd": round(sum(item["fees_usd"] for item in state.trades), 8),
        "total_slippage_cost_usd": round(
            sum(item["slippage_cost_usd"] for item in state.trades), 8
        ),
        **summary,
        "trade_management": {
            "atr_period": management.atr_period,
            "take_profit_atr_multiplier": management.take_profit_atr_multiplier,
            "stop_loss_atr_multiplier": management.stop_loss_atr_multiplier,
            "horizon_candles": management.horizon_candles,
            "trailing_stop_enabled": management.trailing_stop_enabled,
            "trailing_activation_atr": management.trailing_activation_atr,
            "trailing_step_atr": management.trailing_step_atr,
            "trailing_stop_update_policy": ("COMPLETED_BAR_HIGH_NEXT_BAR_EFFECTIVE"),
            "time_exit_candles": management.time_exit_candles,
            "time_exit_mfe_threshold_atr": management.time_exit_mfe_threshold_atr,
            "time_exit_evaluation_policy": "ONCE_AFTER_X_COMPLETED_CANDLES",
            "time_exit_execution_policy": "NEXT_CANDLE_OPEN",
            "trailing_breakeven_policy": "NET_OF_BOTH_SIDE_FEES_AND_SLIPPAGE",
            "barrier_reference": "ACTUAL_NEXT_BAR_ENTRY_FILL",
            "atr_reference": "SIGNAL_BAR_CLOSE_ONLY",
            "same_candle_collision_policy": "STOP_LOSS_WINS",
            "hold_semantics": "KEEP_POSITION_UNTIL_BARRIER_OR_HORIZON",
            "sell_semantics": "CLOSE_LONG_AT_NEXT_BAR_OPEN",
            "short_positions_supported": False,
        },
    }


def evaluate_candidate(
    candidate: dict[str, Any],
    bars: DataFrame,
    config: ExecutionConfig,
    trade_management_config: TradeManagementConfig | None = None,
) -> dict[str, Any]:
    """Replay persisted probabilities with fixed Spot trade management."""
    management = trade_management_config or TradeManagementConfig()
    validate_trade_management_config(management)
    signals = build_forward_signals(
        candidate, confidence_threshold=management.confidence_threshold
    )
    replay = simulate_signals(bars, signals, config, trade_management_config=management)
    return {
        "candidate_id": candidate["candidate_id"],
        "scenario": candidate["scenario"],
        "pair": candidate["pair"],
        "timeframe": candidate["timeframe"],
        "feature_profile": candidate["feature_profile"],
        "weighting_policy": candidate["weighting_policy"],
        "selected_parameters": candidate["selected_parameters"],
        "source_forward_hold_precision": candidate["hold_precision"],
        "source_forward_hold_recall": candidate["hold_recall"],
        "selected_hold_threshold": candidate["selected_hold_threshold"],
        "forward_signal_count": len(signals),
        "raw_signal_counts": {
            signal: sum(item["raw_signal"] == signal for item in signals)
            for signal in LABELS
        },
        **replay,
    }


V4L_CONFIDENCE_SWEEP_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)


def validate_confidence_thresholds(
    confidence_thresholds: list[float] | tuple[float, ...],
) -> tuple[float, ...]:
    """Validate a non-empty, unique confidence sweep without reordering it."""
    if (
        not isinstance(confidence_thresholds, (list, tuple))
        or not confidence_thresholds
    ):
        raise ValueError("economic_confidence_sweep_thresholds_invalid")
    validated: list[float] = []
    seen: set[float] = set()
    for value in confidence_thresholds:
        if isinstance(value, bool):
            raise ValueError("economic_confidence_sweep_threshold_invalid")
        try:
            threshold = _validate_confidence_threshold(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("economic_confidence_sweep_threshold_invalid") from None
        if threshold in seen:
            raise ValueError("economic_confidence_sweep_threshold_duplicate")
        seen.add(threshold)
        validated.append(threshold)
    return tuple(validated)


def evaluate_candidate_confidence_sweep(
    candidate: dict[str, Any],
    bars: DataFrame,
    config: ExecutionConfig,
    base_management: TradeManagementConfig | None = None,
    *,
    confidence_thresholds: list[float] | tuple[float, ...] = (
        V4L_CONFIDENCE_SWEEP_THRESHOLDS
    ),
) -> list[dict[str, Any]]:
    """Replay one persisted candidate independently at each BUY confidence floor.

    Stored probabilities, bars, execution assumptions, and exit geometry remain
    identical across runs. Only the BUY-entry confidence floor changes.
    """
    thresholds = validate_confidence_thresholds(confidence_thresholds)
    base = base_management or TradeManagementConfig()
    validate_trade_management_config(base)
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        management = TradeManagementConfig(
            atr_period=base.atr_period,
            take_profit_atr_multiplier=base.take_profit_atr_multiplier,
            stop_loss_atr_multiplier=base.stop_loss_atr_multiplier,
            horizon_candles=base.horizon_candles,
            confidence_threshold=threshold,
            trailing_stop_enabled=base.trailing_stop_enabled,
            trailing_activation_atr=base.trailing_activation_atr,
            trailing_step_atr=base.trailing_step_atr,
            time_exit_candles=base.time_exit_candles,
            time_exit_mfe_threshold_atr=base.time_exit_mfe_threshold_atr,
        )
        results.append(
            {
                "confidence_threshold": threshold,
                "candidate_result": evaluate_candidate(
                    candidate, bars, config, management
                ),
            }
        )
    return results


V4M_CONFIDENCE_THRESHOLD = 0.60
V4N_CONFIDENCE_THRESHOLD = 0.60
V4N_STOP_LOSS_ATR_MULTIPLIER = 2.0
V4N_TAKE_PROFIT_ATR_MULTIPLIER = 3.0
V4N_TRAILING_ACTIVATION_ATR = 1.0
V4N_TRAILING_STEP_ATR = 1.0
V4O_TIME_EXIT_MFE_THRESHOLD_ATR = 0.5
V4O_TIME_EXIT_CANDLES = (None, 3, 6, 9)
# Each tuple is ordered (stop_loss_atr_multiplier, take_profit_atr_multiplier).
V4M_TP_SL_GEOMETRIES = (
    (2.0, 2.0),
    (2.5, 2.5),
    (2.0, 3.0),
    (1.5, 2.0),
)


def _validate_tp_sl_multiplier(value: Any) -> float:
    """Convert one ATR multiplier and reject non-finite or non-positive values."""
    if isinstance(value, bool):
        raise ValueError("economic_tp_sl_geometry_invalid")
    try:
        multiplier = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("economic_tp_sl_geometry_invalid") from None
    if not np.isfinite(multiplier):
        raise ValueError("economic_tp_sl_geometry_invalid")
    if multiplier <= 0:
        raise ValueError("economic_tp_sl_multipliers_must_be_positive")
    return multiplier


def _normalize_tp_sl_geometry(geometry: Any) -> tuple[float, float]:
    """Normalize one (SL, TP) pair without accepting malformed shapes."""
    if not isinstance(geometry, (list, tuple)) or len(geometry) != 2:
        raise ValueError("economic_tp_sl_geometry_invalid")
    stop_loss = _validate_tp_sl_multiplier(geometry[0])
    take_profit = _validate_tp_sl_multiplier(geometry[1])
    return stop_loss, take_profit


def validate_tp_sl_geometries(
    tp_sl_geometries: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Validate unique positive finite (SL, TP) ATR multiplier pairs."""
    if not isinstance(tp_sl_geometries, (list, tuple)) or not tp_sl_geometries:
        raise ValueError("economic_tp_sl_geometries_invalid")
    validated: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for geometry in tp_sl_geometries:
        pair = _normalize_tp_sl_geometry(geometry)
        if pair in seen:
            raise ValueError("economic_tp_sl_geometry_duplicate")
        seen.add(pair)
        validated.append(pair)
    return tuple(validated)


def evaluate_candidate_tp_sl_sweep(
    candidate: dict[str, Any],
    bars: DataFrame,
    config: ExecutionConfig,
    base_management: TradeManagementConfig | None = None,
    *,
    tp_sl_geometries: list[tuple[float, float]]
    | tuple[tuple[float, float], ...] = V4M_TP_SL_GEOMETRIES,
) -> list[dict[str, Any]]:
    """Replay one persisted candidate over TP/SL geometries at fixed 0.60 confidence.

    Probability rows, market bars, execution costs, ATR period, and vertical
    horizon remain fixed. Each geometry runs an independent Spot simulation.
    """
    geometries = validate_tp_sl_geometries(tp_sl_geometries)
    base = base_management or TradeManagementConfig(
        confidence_threshold=V4M_CONFIDENCE_THRESHOLD
    )
    validate_trade_management_config(base)
    if base.confidence_threshold != V4M_CONFIDENCE_THRESHOLD:
        raise ValueError("v4m_confidence_threshold_must_be_0_60")
    results: list[dict[str, Any]] = []
    for stop_loss, take_profit in geometries:
        management = TradeManagementConfig(
            atr_period=base.atr_period,
            take_profit_atr_multiplier=take_profit,
            stop_loss_atr_multiplier=stop_loss,
            horizon_candles=base.horizon_candles,
            confidence_threshold=V4M_CONFIDENCE_THRESHOLD,
            trailing_stop_enabled=base.trailing_stop_enabled,
            trailing_activation_atr=base.trailing_activation_atr,
            trailing_step_atr=base.trailing_step_atr,
            time_exit_candles=base.time_exit_candles,
            time_exit_mfe_threshold_atr=base.time_exit_mfe_threshold_atr,
        )
        results.append(
            {
                "stop_loss_atr_multiplier": stop_loss,
                "take_profit_atr_multiplier": take_profit,
                "candidate_result": evaluate_candidate(
                    candidate, bars, config, management
                ),
            }
        )
    return results


def validate_time_exit_candles(
    time_exit_candles: list[int | None] | tuple[int | None, ...],
) -> tuple[int | None, ...]:
    """Validate a non-empty ordered collection of unique exit horizons."""
    if not isinstance(time_exit_candles, (list, tuple)) or not time_exit_candles:
        raise ValueError("economic_time_exit_sweep_values_invalid")
    validated: list[int | None] = []
    seen: set[int | None] = set()
    for value in time_exit_candles:
        if value is not None:
            _require_positive_integer(value, "economic_time_exit_candles_invalid")
        if value in seen:
            raise ValueError("economic_time_exit_sweep_duplicate")
        seen.add(value)
        validated.append(value)
    return tuple(validated)


def evaluate_candidate_time_exit_sweep(
    candidate: dict[str, Any],
    bars: DataFrame,
    config: ExecutionConfig,
    base_management: TradeManagementConfig | None = None,
    *,
    time_exit_candles: list[int | None] | tuple[int | None, ...] = (
        V4O_TIME_EXIT_CANDLES
    ),
) -> list[dict[str, Any]]:
    """Replay stored probabilities at baseline and fixed dynamic time exits."""
    options = validate_time_exit_candles(time_exit_candles)
    base = base_management or TradeManagementConfig(
        confidence_threshold=V4N_CONFIDENCE_THRESHOLD,
        trailing_stop_enabled=False,
        time_exit_mfe_threshold_atr=V4O_TIME_EXIT_MFE_THRESHOLD_ATR,
    )
    validate_trade_management_config(base)
    if base.trailing_stop_enabled:
        raise ValueError("v4o_trailing_stop_must_be_disabled")
    if base.confidence_threshold != V4N_CONFIDENCE_THRESHOLD:
        raise ValueError("v4o_confidence_threshold_must_be_0_60")
    results: list[dict[str, Any]] = []
    for candles in options:
        management = TradeManagementConfig(
            atr_period=base.atr_period,
            take_profit_atr_multiplier=base.take_profit_atr_multiplier,
            stop_loss_atr_multiplier=base.stop_loss_atr_multiplier,
            horizon_candles=base.horizon_candles,
            confidence_threshold=base.confidence_threshold,
            trailing_stop_enabled=False,
            trailing_activation_atr=base.trailing_activation_atr,
            trailing_step_atr=base.trailing_step_atr,
            time_exit_candles=candles,
            time_exit_mfe_threshold_atr=base.time_exit_mfe_threshold_atr,
        )
        results.append(
            {
                "time_exit_candles": candles,
                "candidate_result": evaluate_candidate(
                    candidate, bars, config, management
                ),
            }
        )
    return results


def render_economic_markdown(report: dict[str, Any]) -> str:
    """Render the auditable per-model economic results into GitHub Step Summary."""
    lines = [
        "## AliX V4-I Economic Evaluation",
        "",
        f"Source V4-H SHA: {report['source_v4h_commit_sha']}",
        f"V4-I commit SHA: {report['commit_sha']}",
        "",
        "Replay only: persisted V4-H Forward-Validation probabilities; no model fitting or inference.",
        "",
        "| Pair | TF | Profile | Weighting | HOLD P | HOLD R | Profit Factor | Max DD | Net Return | Trades | Gate |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["candidates"]:
        metrics = item["metrics"]
        pf = (
            "INF"
            if metrics["profit_factor_infinite_no_losses"]
            else f"{metrics['profit_factor_net_of_costs']:.4f}"
            if metrics["profit_factor_net_of_costs"] is not None
            else "N/A"
        )
        lines.append(
            f"| {item['pair']} | {item['timeframe']} | {item['feature_profile']} "
            f"| {item['weighting_policy']} | {item['source_forward_hold_precision']:.4f} "
            f"| {item['source_forward_hold_recall']:.4f} | {pf} "
            f"| {metrics['max_drawdown_percent']:.2f}% "
            f"| {metrics['net_return_percent']:.2f}% | {metrics['trade_count']} "
            f"| {'PASS' if metrics['economic_candidate_gate_passed'] else 'NO-GO'} |"
        )
    lines.extend(
        [
            "",
            f"Candidates passing PF > 1.5 and Max DD < 15%: {report['economic_acceptance']['passed_candidate_count']}/"
            f"{report['economic_acceptance']['candidate_count']}.",
            "",
            "Execution: next-bar open; BUY=long, SELL=short, HOLD=close/flat; 1.0x equity, no pyramiding.",
            "Costs: 25 bps fee and 20 bps slippage per side. No funding/borrow, stop-loss, take-profit, or trailing stop.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0.",
            "This is a replay of the same Forward-Validation interval used to select V4-H candidates, not an independent out-of-sample promotion test.",
        ]
    )
    return "\n".join(lines) + "\n"
