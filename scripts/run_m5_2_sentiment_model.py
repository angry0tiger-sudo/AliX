"""Evaluate the frozen M5.2 BTCUSDT control and sentiment models in GitHub Actions."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
import sklearn  # type: ignore[import-untyped]

from scripts.align_m5_sentiment import align_records

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/m5_2/sentiment_model_preregistration.json"
DATA_LOCK_PATH = ROOT / "data/nobitex/prehack/m5_2_btcusdt_1h_data_lock.json"
OHLCV_PATH = ROOT / "data/nobitex/prehack/nobitex-btcusdt-1h.csv"
OHLCV_MANIFEST_PATH = ROOT / "data/nobitex/prehack/nobitex-btcusdt-1h.manifest.json"

PREREG_SHA256 = "851d6c4b39c41cacec8ccd712cf17bb40dd42dd45f2bb8813920a7f2ebe481ba"
DATA_LOCK_SHA256 = "e66616ee3c13e6ce32f94577c4f46480390eea30c774ac6987e8d519d51d4acf"
OHLCV_SHA256 = "0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e"
OHLCV_MANIFEST_SHA256 = (
    "b50fea25f30d282c2eafcb262751367e11d6490eda0d463de5926b660eaf1a26"
)
SENTIMENT_RAW_SHA256 = (
    "4adba36f7dd95792520e078f7578c316c23e80a7b8a0ace81df490bf06c740df"
)
SENTIMENT_NORMALIZED_SHA256 = (
    "e0578509cf3f8a6eeac39ddbd2edeab876e47027d759558e120c6f8b3a9f7f61"
)

CONTROL_FEATURES = (
    "log_return_1",
    "log_return_4",
    "log_return_24",
    "realized_volatility_6",
    "realized_volatility_24",
    "range_fraction_1",
    "atr_fraction_14",
    "log_volume_change_1",
)
TREATMENT_FEATURES = CONTROL_FEATURES + ("fear_greed_value",)
HORIZONS = (1, 4)
ARMS = ("control", "sentiment")
COST_NAMES = ("Cost-50", "Cost-90")
EXPECTED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Candle:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class FoldPlan:
    fold_id: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    mark_indices: tuple[int, ...]
    first_test_index: int
    last_test_index: int
    purged_train_rows: int
    excluded_test_rows: int


@dataclass(frozen=True)
class Trade:
    decision_index: int
    entry_index: int
    exit_index: int
    notional: float
    pnl: float


@dataclass(frozen=True)
class PortfolioOutcome:
    metrics: dict[str, Any]
    nav: tuple[float, ...]
    pnl: float
    trades: tuple[Trade, ...]


@dataclass(frozen=True)
class FoldOutput:
    summary: dict[str, Any]
    arm_outcomes: dict[str, dict[str, PortfolioOutcome]]
    baseline_outcomes: dict[str, dict[str, PortfolioOutcome]]
    predictions: tuple[dict[str, Any], ...]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, error_code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError(error_code) from None
    if not isinstance(payload, dict):
        raise ValueError(error_code)
    return cast(dict[str, Any], payload)


def _validate_market_scope(config: dict[str, Any]) -> None:
    scope = config["scope"]
    targets = config["targets"]
    if scope["venue"] != "Nobitex" or scope["market"] != "BTCUSDT":
        raise ValueError("m5_2_market_scope_mismatch")
    if scope["market_type"] != "Spot" or scope["quote_currency"] != "USDT":
        raise ValueError("m5_2_quote_scope_mismatch")
    if tuple(targets["horizons"]) != HORIZONS:
        raise ValueError("m5_2_horizon_lock_mismatch")
    if scope["decision_timeframe"] != "1h completed OHLCV candles":
        raise ValueError("m5_2_decision_timeframe_mismatch")


def _validate_feature_and_model_lock(config: dict[str, Any]) -> None:
    feature_spec = config["feature_specification"]
    if tuple(feature_spec["feature_names_ohlcv_only_control"]) != CONTROL_FEATURES:
        raise ValueError("m5_2_control_feature_lock_mismatch")
    if tuple(feature_spec["feature_names_sentiment_augmented"]) != TREATMENT_FEATURES:
        raise ValueError("m5_2_treatment_feature_lock_mismatch")
    protocol = config["walk_forward_protocol"]
    if protocol["fold_count"] != 4 or len(protocol["folds"]) != 4:
        raise ValueError("m5_2_fold_count_mismatch")
    if protocol["scheme"] != "expanding_window_on_five_contiguous_chronological_blocks":
        raise ValueError("m5_2_fold_scheme_mismatch")
    model = config["model"]
    if model["estimator"] != "sklearn.linear_model.LogisticRegression":
        raise ValueError("m5_2_estimator_mismatch")
    if model["pipeline"] != [
        "sklearn.preprocessing.StandardScaler",
        "sklearn.linear_model.LogisticRegression",
    ]:
        raise ValueError("m5_2_pipeline_mismatch")
    if model["parameters"] != {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "fit_intercept": True,
        "class_weight": None,
        "max_iter": 1000,
        "tol": 0.0001,
        "random_state": 42,
    }:
        raise ValueError("m5_2_model_parameters_mismatch")
    if model["signal"]["long_entry_probability_threshold"] != 0.5:
        raise ValueError("m5_2_signal_threshold_mismatch")


def _validate_costs_and_gates(config: dict[str, Any]) -> None:
    expected = {"Cost-50": (50, 0.005), "Cost-90": (90, 0.009)}
    costs = config["portfolio_evaluation"]["cost_scenarios"]
    for name, values in expected.items():
        item = costs[name]
        if (item["round_trip_cost_bps"], item["fraction"]) != values:
            raise ValueError(f"m5_2_cost_lock_mismatch:{name}")
    gates = config["acceptance_gates"]
    if gates["net_profit_factor_min"] != 1.10:
        raise ValueError("m5_2_netpf_gate_mismatch")
    if gates["completed_oos_trades_min"] != 100:
        raise ValueError("m5_2_trade_count_gate_mismatch")
    if gates["max_drawdown_fraction_max"] != 0.15:
        raise ValueError("m5_2_maxdd_gate_mismatch")
    if gates["minimum_valid_folds"] != 3 or gates["minimum_stable_folds"] != 3:
        raise ValueError("m5_2_fold_gate_mismatch")


def _validate_safety_lock(config: dict[str, Any]) -> None:
    safety = config["safety"]
    if safety["holdout_access"] != "PROHIBITED":
        raise ValueError("m5_2_holdout_lock_changed")
    if safety["posthack_data_access"] != "PROHIBITED":
        raise ValueError("m5_2_posthack_lock_changed")
    if safety["private_api_calls"] != 0 or safety["real_order_attempts"] != 0:
        raise ValueError("m5_2_safety_lock_changed")
    if safety["local_execution"] is not False:
        raise ValueError("m5_2_local_execution_lock_changed")
    if safety["Champion"] is not None or safety["order_permission"] is not False:
        raise ValueError("m5_2_promotion_lock_changed")
    if safety["trading_authorization"] != "NO-GO":
        raise ValueError("m5_2_trading_lock_changed")


def load_preregistration() -> tuple[dict[str, Any], str]:
    raw = CONFIG_PATH.read_bytes()
    digest = sha256_bytes(raw)
    if digest != PREREG_SHA256:
        raise ValueError("m5_2_preregistration_sha256_mismatch")
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("m5_2_preregistration_json_invalid") from None
    if not isinstance(config, dict):
        raise ValueError("m5_2_preregistration_root_invalid")
    config = cast(dict[str, Any], config)
    if config["preregistration"]["id"] != "ALIX-M5-2-SENTIMENT-G1":
        raise ValueError("m5_2_preregistration_id_mismatch")
    if config["preregistration"]["status"] != "PREREGISTERED_LOCKED":
        raise ValueError("m5_2_preregistration_status_mismatch")
    _validate_market_scope(config)
    _validate_feature_and_model_lock(config)
    _validate_costs_and_gates(config)
    _validate_safety_lock(config)
    return config, digest


def require_owner_authorization() -> None:
    if os.environ.get("ALIX_M52_OWNER_AUTHORIZATION") != "true":
        raise PermissionError("m5_2_owner_authorization_missing")
    if os.environ.get("ALIx_DRY_RUN") != "true":
        raise PermissionError("m5_2_dry_run_required")
    if os.environ.get("ALIx_ALLOW_LIVE_TRADING") != "false":
        raise PermissionError("m5_2_live_trading_must_be_false")


def _check_data_lock(lock: dict[str, Any]) -> None:
    expected = (
        lock["venue"] == "Nobitex",
        lock["market"] == "BTCUSDT",
        lock["pair"] == "BTC/USDT",
        lock["market_type"] == "Spot",
        lock["timeframe"] == "1h",
        lock["data_scope"] == "PRE_HACK_ONLY",
        lock["row_count"] == 31801,
        lock["raw_sha256"] == OHLCV_SHA256,
        lock["source_manifest_sha256"] == OHLCV_MANIFEST_SHA256,
        lock["gap_report"]["expected_intervals"] == 31800,
        lock["gap_report"]["missing_intervals"] == 0,
        lock["gap_report"]["off_grid_intervals"] == 0,
        lock["gap_report"]["status"] == "PASS",
        lock["new_data_acquisition"] is False,
        lock["holdout_bytes_read"] is False,
        lock["posthack_data_read"] is False,
        lock["private_api_calls"] == 0,
        lock["real_order_attempts"] == 0,
        lock["paper_orders"] == 0,
        lock["Champion"] is None,
        lock["order_permission"] is False,
        lock["trading_authorization"] == "NO-GO",
    )
    if not all(expected):
        raise ValueError("m5_2_data_lock_contract_invalid")


def _check_ohlcv_manifest(manifest: dict[str, Any]) -> None:
    expected = (
        manifest["dataset_window"] == "PRE_HACK_NOBITEX_ONLY",
        manifest["symbol"] == "BTCUSDT",
        manifest["pair"] == "BTC/USDT",
        manifest["interval"] == "1h",
        manifest["row_count"] == 31801,
        manifest["expected_candles"] == 31801,
        manifest["missing_intervals"] == 0,
        manifest["csv_sha256"] == OHLCV_SHA256,
        manifest["public_api_only"] is True,
        manifest["credentials_used"] is False,
        manifest["private_api_calls"] == 0,
        manifest["real_order_attempts"] == 0,
        manifest["exchange_calls"] == 0,
    )
    if not all(expected):
        raise ValueError("m5_2_source_manifest_contract_invalid")


def _parse_datetime(raw: str) -> datetime:
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    result = datetime.fromisoformat(normalized)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("m5_2_timestamp_missing_timezone")
    return result.astimezone(timezone.utc)


def _parse_candle(row: dict[str, str]) -> Candle:
    start = _parse_datetime(row["date"])
    values = [float(row[key]) for key in ("open", "high", "low", "close", "volume")]
    opened, high, low, close, volume = values
    if not all(math.isfinite(item) for item in values):
        raise ValueError("m5_2_ohlcv_non_finite")
    if min(opened, high, low, close) <= 0.0 or volume < 0.0:
        raise ValueError("m5_2_ohlcv_value_out_of_range")
    if high < max(opened, low, close) or low > min(opened, high, close):
        raise ValueError("m5_2_ohlc_invariant_failed")
    return Candle(start, opened, high, low, close, volume)


def _load_ohlcv_bindings() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_bytes(DATA_LOCK_PATH.read_bytes()) != DATA_LOCK_SHA256:
        raise ValueError("m5_2_data_lock_sha256_mismatch")
    if sha256_bytes(OHLCV_PATH.read_bytes()) != OHLCV_SHA256:
        raise ValueError("m5_2_ohlcv_sha256_mismatch")
    if sha256_bytes(OHLCV_MANIFEST_PATH.read_bytes()) != OHLCV_MANIFEST_SHA256:
        raise ValueError("m5_2_source_manifest_sha256_mismatch")
    lock = read_json(DATA_LOCK_PATH, "m5_2_data_lock_invalid")
    manifest = read_json(OHLCV_MANIFEST_PATH, "m5_2_source_manifest_invalid")
    _check_data_lock(lock)
    _check_ohlcv_manifest(manifest)
    if (
        lock["start_utc"] != manifest["start_utc"]
        or lock["end_utc"] != manifest["end_utc"]
    ):
        raise ValueError("m5_2_data_lock_window_mismatch")
    return lock, manifest


def _read_ohlcv_candles() -> list[Candle]:
    candles: list[Candle] = []
    try:
        with OHLCV_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
                raise ValueError("m5_2_ohlcv_columns_invalid")
            for row in reader:
                candles.append(_parse_candle(cast(dict[str, str], row)))
    except (OSError, csv.Error, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"m5_2_ohlcv_csv_invalid:{type(exc).__name__}") from None
    if len(candles) != 31801:
        raise ValueError("m5_2_ohlcv_row_count_invalid")
    return candles


def _validate_ohlcv_timeline(
    candles: Sequence[Candle], manifest: dict[str, Any]
) -> None:
    for previous, current in zip(candles, candles[1:]):
        if (current.start - previous.start).total_seconds() != 3600:
            raise ValueError("m5_2_ohlcv_hourly_grid_or_gap_invalid")
    first = candles[0].start.isoformat().replace("+00:00", "Z")
    last = candles[-1].start.isoformat().replace("+00:00", "Z")
    if first != manifest["start_utc"]:
        raise ValueError("m5_2_ohlcv_start_mismatch")
    if last != manifest["end_utc"]:
        raise ValueError("m5_2_ohlcv_end_mismatch")


def load_verified_ohlcv() -> tuple[list[Candle], dict[str, Any]]:
    _lock, manifest = _load_ohlcv_bindings()
    candles = _read_ohlcv_candles()
    _validate_ohlcv_timeline(candles, manifest)
    return candles, manifest


def _validate_sentiment_manifest(
    manifest: dict[str, Any], raw_hash: str, normalized_hash: str
) -> None:
    expected = (
        manifest["raw_sha256"] == raw_hash,
        manifest["normalized_sha256"] == normalized_hash,
        manifest["record_count"] == 3218,
        manifest["provider_supplies_historical_available_at"] is False,
        manifest["availability_policy"]
        == "observation_timestamp_plus_24h_conservative_proxy",
    )
    if not all(expected):
        raise ValueError("m5_2_sentiment_manifest_contract_invalid")


def _parse_sentiment_records(
    normalized_raw: bytes,
) -> list[Mapping[str, object]]:
    try:
        payload = json.loads(normalized_raw)
    except json.JSONDecodeError:
        raise ValueError("m5_2_sentiment_json_invalid") from None
    if not isinstance(payload, dict):
        raise ValueError("m5_2_sentiment_root_invalid")
    if payload.get("schema_version") != "alix.m5.alternative-me-fng.v1":
        raise ValueError("m5_2_sentiment_schema_mismatch")
    if payload.get("provider_id") != "alternative_me_fear_greed":
        raise ValueError("m5_2_sentiment_provider_mismatch")
    if payload.get("target_market") != "Nobitex Spot/USDT BTCUSDT":
        raise ValueError("m5_2_sentiment_target_market_mismatch")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != 3218:
        raise ValueError("m5_2_sentiment_record_count_invalid")
    return _validate_sentiment_records(raw_records)


def _validate_sentiment_records(
    raw_records: list[object],
) -> list[Mapping[str, object]]:
    records: list[Mapping[str, object]] = []
    for item in raw_records:
        if not isinstance(item, dict):
            raise ValueError("m5_2_sentiment_record_invalid")
        scope = item.get("market_scope")
        availability = item.get("available_at_basis")
        if scope != "global_btc_sentiment_not_nobitex_pair_specific":
            raise ValueError("m5_2_sentiment_market_scope_invalid")
        if availability != "observation_timestamp_plus_24h_conservative_proxy":
            raise ValueError("m5_2_sentiment_record_availability_invalid")
        records.append(cast(Mapping[str, object], item))
    return records


def _resolve_repository_relative_path(value: object) -> Path:
    relative_path = Path(str(value))
    if relative_path.is_absolute() or not str(value).strip():
        raise ValueError("m5_2_sentiment_path_must_be_repository_relative")
    root = ROOT.resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("m5_2_sentiment_path_outside_repository") from None
    return resolved


def load_verified_sentiment(
    config: dict[str, Any],
) -> tuple[list[Mapping[str, object]], dict[str, Any]]:
    source = config["sentiment_source"]
    manifest_path = _resolve_repository_relative_path(source["snapshot_manifest"])
    raw_path = _resolve_repository_relative_path(source["raw_file"])
    normalized_path = _resolve_repository_relative_path(source["normalized_file"])
    if sha256_bytes(raw_path.read_bytes()) != SENTIMENT_RAW_SHA256:
        raise ValueError("m5_2_sentiment_raw_sha256_mismatch")
    normalized_raw = normalized_path.read_bytes()
    if sha256_bytes(normalized_raw) != SENTIMENT_NORMALIZED_SHA256:
        raise ValueError("m5_2_sentiment_normalized_sha256_mismatch")
    manifest = read_json(manifest_path, "m5_2_sentiment_manifest_invalid")
    _validate_sentiment_manifest(
        manifest, SENTIMENT_RAW_SHA256, SENTIMENT_NORMALIZED_SHA256
    )
    records = _parse_sentiment_records(normalized_raw)
    return records, manifest


def build_price_features(candles: Sequence[Candle]) -> NDArray[Any]:
    count = len(candles)
    close = np.asarray([row.close for row in candles], dtype=np.float64)
    high = np.asarray([row.high for row in candles], dtype=np.float64)
    low = np.asarray([row.low for row in candles], dtype=np.float64)
    volume = np.asarray([row.volume for row in candles], dtype=np.float64)
    log_close = np.log(close)
    returns = np.full(count, np.nan, dtype=np.float64)
    returns[1:] = np.diff(log_close)
    tr = high - low
    tr[1:] = np.maximum.reduce(
        (
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        )
    )
    atr = np.full(count, np.nan, dtype=np.float64)
    if count >= 14:
        atr[13] = float(np.mean(tr[:14]))
        for index in range(14, count):
            atr[index] = (13.0 * atr[index - 1] + tr[index]) / 14.0
    matrix = np.full((count, 8), np.nan, dtype=np.float64)
    matrix[:, 0] = returns
    matrix[4:, 1] = log_close[4:] - log_close[:-4]
    matrix[24:, 2] = log_close[24:] - log_close[:-24]
    for index in range(5, count):
        matrix[index, 3] = float(np.std(returns[index - 5 : index + 1], ddof=0))
    for index in range(23, count):
        matrix[index, 4] = float(np.std(returns[index - 23 : index + 1], ddof=0))
    matrix[:, 5] = (high - low) / close
    matrix[:, 6] = atr / close
    volume_change = np.full(count, np.nan, dtype=np.float64)
    volume_change[1:] = np.diff(np.log1p(volume))
    matrix[:, 7] = volume_change
    return matrix


def align_sentiment(
    records: Sequence[Mapping[str, object]],
    decision_times: Sequence[str],
) -> tuple[NDArray[Any], list[str | None], list[str | None], NDArray[Any]]:
    aligned = align_records(records, decision_times, timeframe="1h")
    values = np.full(len(aligned), np.nan, dtype=np.float64)
    available: list[str | None] = []
    observations: list[str | None] = []
    ages = np.full(len(aligned), np.nan, dtype=np.float64)
    for index, item in enumerate(aligned):
        if item["status"] == "no_prior_available_record":
            available.append(None)
            observations.append(None)
            continue
        value = item["feature_value"]
        available_at = item["source_available_at"]
        observation = item["source_observation_time_utc"]
        age = item["source_age_seconds"]
        if not isinstance(value, int) or not isinstance(available_at, str):
            raise ValueError("m5_2_aligned_sentiment_invalid")
        if not isinstance(observation, str) or not isinstance(age, int) or age < 0:
            raise ValueError("m5_2_aligned_sentiment_provenance_invalid")
        decision_time = item["decision_time"]
        if not isinstance(decision_time, str):
            raise ValueError("m5_2_aligned_decision_time_invalid")
        if available_at > decision_time:
            raise ValueError("m5_2_future_sentiment_leakage")
        values[index] = float(value)
        available.append(available_at)
        observations.append(observation)
        ages[index] = float(age)
    return values, available, observations, ages


def _make_fold_plan(
    fold_index: int,
    train_candidates: Sequence[int],
    test_candidates: Sequence[int],
    label_valid: Sequence[bool],
    horizon: int,
) -> FoldPlan:
    first_test = test_candidates[0]
    last_test = test_candidates[-1]
    train = tuple(
        index
        for index in train_candidates
        if label_valid[index] and index + horizon < first_test
    )
    test = tuple(
        index
        for index in test_candidates
        if label_valid[index] and index + horizon <= last_test
    )
    purged = sum(
        1
        for index in train_candidates
        if label_valid[index] and index + horizon >= first_test
    )
    return FoldPlan(
        fold_id=f"WF-{fold_index + 1:02d}",
        train_indices=train,
        test_indices=test,
        mark_indices=tuple(range(first_test, last_test + 1)),
        first_test_index=first_test,
        last_test_index=last_test,
        purged_train_rows=purged,
        excluded_test_rows=len(test_candidates) - len(test),
    )


def build_fold_plans(
    eligible: Sequence[int], label_valid: Sequence[bool], horizon: int
) -> tuple[FoldPlan, ...]:
    if horizon not in HORIZONS:
        raise ValueError("m5_2_horizon_not_registered")
    if len(eligible) < 5:
        raise ValueError("m5_2_insufficient_eligible_rows_for_blocks")
    if any(left >= right for left, right in zip(eligible, eligible[1:])):
        raise ValueError("m5_2_eligible_rows_not_strictly_ordered")
    count = len(eligible)
    edges = [index * count // 5 for index in range(6)]
    plans: list[FoldPlan] = []
    for fold_index in range(4):
        train_candidates = eligible[: edges[fold_index + 1]]
        test_candidates = eligible[edges[fold_index + 1] : edges[fold_index + 2]]
        if not test_candidates:
            raise ValueError("m5_2_empty_test_block")
        plans.append(
            _make_fold_plan(
                fold_index,
                train_candidates,
                test_candidates,
                label_valid,
                horizon,
            )
        )
    return tuple(plans)


def build_model(config: dict[str, Any]) -> Pipeline:
    params = config["model"]["parameters"]
    classifier = LogisticRegression(
        penalty=str(params["penalty"]),
        C=float(params["C"]),
        solver=str(params["solver"]),
        fit_intercept=bool(params["fit_intercept"]),
        class_weight=params["class_weight"],
        max_iter=int(params["max_iter"]),
        tol=float(params["tol"]),
        random_state=int(params["random_state"]),
    )
    return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])


def net_profit_factor(trades: Sequence[Trade]) -> float | None:
    wins = sum(row.pnl for row in trades if row.pnl > 0.0)
    losses = -sum(row.pnl for row in trades if row.pnl < 0.0)
    if losses <= 0.0:
        return None
    return float(wins / losses)


def max_drawdown(nav: Sequence[float]) -> float:
    peak = 1.0
    worst = 0.0
    for value in nav:
        peak = max(peak, value)
        if peak > 0.0:
            worst = max(worst, (peak - value) / peak)
    return float(worst)


def _build_trades(
    candles: Sequence[Candle],
    decisions: Sequence[int],
    signals: Sequence[bool],
    horizon: int,
    cost: float,
    last_mark: int,
) -> list[Trade]:
    notional = 1.0 / float(horizon)
    trades: list[Trade] = []
    for decision, signal in zip(decisions, signals):
        if signal:
            entry = decision + 1
            exit_index = decision + horizon
            if exit_index > last_mark:
                raise ValueError("m5_2_trade_exit_crosses_test_fold")
            gross = candles[exit_index].close / candles[entry].open - 1.0
            trades.append(
                Trade(decision, entry, exit_index, notional, notional * (gross - cost))
            )
    trades.sort(key=lambda row: (row.entry_index, row.exit_index))
    return trades


def _mark_portfolio(
    candles: Sequence[Candle],
    trades: Sequence[Trade],
    marks: Sequence[int],
    cost: float,
) -> tuple[list[float], int]:
    cursor = 0
    active: list[Trade] = []
    nav = [1.0]
    concurrent = 0
    for mark in marks:
        while cursor < len(trades) and trades[cursor].entry_index <= mark:
            active.append(trades[cursor])
            cursor += 1
        concurrent = max(concurrent, len(active))
        remaining, unrealized = _mark_active_trades(candles, active, mark, cost)
        realized = sum(row.pnl for row in trades[:cursor] if row.exit_index <= mark)
        active = remaining
        nav.append(1.0 + realized + unrealized)
    if active or cursor != len(trades):
        raise ValueError("m5_2_unclosed_fold_position")
    return nav, concurrent


def _mark_active_trades(
    candles: Sequence[Candle],
    active: Sequence[Trade],
    mark: int,
    cost: float,
) -> tuple[list[Trade], float]:
    remaining: list[Trade] = []
    unrealized = 0.0
    for trade in active:
        if trade.exit_index > mark:
            gross_mark = candles[mark].close / candles[trade.entry_index].open - 1.0
            unrealized += trade.notional * (gross_mark - cost)
            remaining.append(trade)
    return remaining, unrealized


def simulate_portfolio(
    candles: Sequence[Candle],
    decisions: Sequence[int],
    signals: Sequence[bool],
    horizon: int,
    cost: float,
    marks: Sequence[int],
) -> PortfolioOutcome:
    if len(decisions) != len(signals):
        raise ValueError("m5_2_signal_count_mismatch")
    if not marks:
        return _empty_portfolio_outcome()
    trades = _build_trades(candles, decisions, signals, horizon, cost, marks[-1])
    nav, concurrent = _mark_portfolio(candles, trades, marks, cost)
    pnl = sum(row.pnl for row in trades)
    metrics = _portfolio_metrics(trades, nav, pnl, concurrent, horizon)
    return PortfolioOutcome(metrics, tuple(nav), float(pnl), tuple(trades))


def _empty_portfolio_outcome() -> PortfolioOutcome:
    return PortfolioOutcome(
        {
            "completed_oos_trades": 0,
            "total_net_pnl_fraction_of_initial_equity": 0.0,
            "net_profit_factor": None,
            "max_drawdown": 0.0,
            "maximum_concurrent_cohorts": 0,
            "turnover_fraction_of_initial_equity": 0.0,
        },
        (1.0,),
        0.0,
        (),
    )


def _portfolio_metrics(
    trades: Sequence[Trade],
    nav: Sequence[float],
    pnl: float,
    concurrent: int,
    horizon: int,
) -> dict[str, Any]:
    if concurrent > horizon:
        raise ValueError("m5_2_position_concurrency_cap_breached")
    notional = 1.0 / float(horizon)
    return {
        "completed_oos_trades": len(trades),
        "total_net_pnl_fraction_of_initial_equity": float(pnl),
        "net_profit_factor": net_profit_factor(trades),
        "max_drawdown": max_drawdown(nav),
        "maximum_concurrent_cohorts": concurrent,
        "turnover_fraction_of_initial_equity": float(len(trades) * notional),
    }


def aggregate_outcomes(outcomes: Sequence[PortfolioOutcome]) -> dict[str, Any]:
    trades = tuple(trade for outcome in outcomes for trade in outcome.trades)
    nav = [1.0]
    carried = 0.0
    for outcome in outcomes:
        nav.extend(value + carried for value in outcome.nav[1:])
        carried += outcome.pnl
    return {
        "completed_oos_trades": len(trades),
        "total_net_pnl_fraction_of_initial_equity": float(
            sum(row.pnl for row in trades)
        ),
        "net_profit_factor": net_profit_factor(trades),
        "max_drawdown": max_drawdown(nav),
        "turnover_fraction_of_initial_equity": float(
            sum(row.notional for row in trades)
        ),
        "maximum_concurrent_cohorts": max(
            (int(row.metrics["maximum_concurrent_cohorts"]) for row in outcomes),
            default=0,
        ),
    }


def is_valid_fold(outcome: PortfolioOutcome) -> bool:
    return int(outcome.metrics["completed_oos_trades"]) >= 20


def is_stable_fold(outcome: PortfolioOutcome) -> bool:
    pf = outcome.metrics["net_profit_factor"]
    return (
        is_valid_fold(outcome)
        and outcome.pnl > 0.0
        and pf is not None
        and float(pf) > 1.0
    )


def check_absolute_gates(
    pooled: dict[str, Any], valid_folds: int, stable_folds: int, config: dict[str, Any]
) -> dict[str, bool]:
    gates = config["acceptance_gates"]
    pf = pooled["net_profit_factor"]
    return {
        "net_profit_factor_gte_1_10": pf is not None
        and float(pf) >= float(gates["net_profit_factor_min"]),
        "completed_oos_trades_gte_100": int(pooled["completed_oos_trades"])
        >= int(gates["completed_oos_trades_min"]),
        "max_drawdown_lte_15_percent": float(pooled["max_drawdown"])
        <= float(gates["max_drawdown_fraction_max"]),
        "at_least_3_valid_folds": valid_folds >= int(gates["minimum_valid_folds"]),
        "at_least_3_stable_folds": stable_folds >= int(gates["minimum_stable_folds"]),
    }


def cost_summary(
    outcomes: Sequence[PortfolioOutcome], config: dict[str, Any]
) -> dict[str, Any]:
    pooled = aggregate_outcomes(outcomes)
    valid = sum(1 for row in outcomes if is_valid_fold(row))
    stable = sum(1 for row in outcomes if is_stable_fold(row))
    gates = check_absolute_gates(pooled, valid, stable, config)
    insufficient = pooled["completed_oos_trades"] < 100 or valid < 3
    if insufficient:
        verdict = "INSUFFICIENT_SAMPLE / NO-GO"
    elif not all(gates.values()):
        verdict = "NO-GO"
    else:
        verdict = "ABSOLUTE_GATES_PASS_DSR_PENDING"
    return {
        "pooled_oos": pooled,
        "valid_folds": valid,
        "stable_folds": stable,
        "acceptance_gates": gates,
        "absolute_gates_passed": all(gates.values()),
        "verdict_before_dsr": verdict,
        "fold_metrics": [row.metrics for row in outcomes],
    }


def _majority_signal(labels: NDArray[Any], train: Sequence[int]) -> bool:
    if not train:
        return False
    counts = Counter(int(labels[index]) for index in train)
    return counts.get(1, 0) > counts.get(0, 0)


def _empty_metrics(
    candles: Sequence[Candle], plan: FoldPlan, horizon: int, cost: float
) -> PortfolioOutcome:
    return simulate_portfolio(candles, (), (), horizon, cost, plan.mark_indices)


def _fold_training_state(
    labels: NDArray[Any], plan: FoldPlan, config: dict[str, Any]
) -> tuple[list[int], list[int], NDArray[Any], Counter[int], bool, bool]:
    train = list(plan.train_indices)
    test = list(plan.test_indices)
    y_train = labels[train]
    counts = Counter(int(value) for value in y_train.tolist())
    rules = config["walk_forward_protocol"]["training_rules"]
    enough = (
        len(train) >= int(rules["minimum_training_rows_per_horizon"])
        and counts.get(0, 0) >= int(rules["minimum_training_rows_per_class"])
        and counts.get(1, 0) >= int(rules["minimum_training_rows_per_class"])
        and bool(test)
    )
    majority = _majority_signal(labels, train)
    return train, test, y_train, counts, enough, majority


def _trial_id_for_arm(arm: str, horizon: int) -> str:
    if arm == "control":
        return f"NOBITEX-M5-2-{horizon}h-OHLCV-LR-BASELINE-01"
    return f"NOBITEX-M5-2-{horizon}h-SENTIMENT-LR-AUGMENTED-01"


def _prediction_row(
    trial_id: str,
    arm: str,
    horizon: int,
    fold_id: str,
    row_index: int,
    probability: float,
    candles: Sequence[Candle],
    labels: NDArray[Any],
    gross: NDArray[Any],
    decision_times: Sequence[str],
    fng_values: NDArray[Any],
    fng_available: Sequence[str | None],
    fng_observations: Sequence[str | None],
    fng_ages: NDArray[Any],
) -> dict[str, Any]:
    exit_time = candles[row_index + horizon].start + timedelta(hours=1)
    return {
        "trial_id": trial_id,
        "horizon_hours": horizon,
        "arm": arm,
        "fold_id": fold_id,
        "decision_time_utc": decision_times[row_index],
        "entry_time_utc": decision_times[row_index],
        "exit_time_utc": exit_time.isoformat().replace("+00:00", "Z"),
        "label": int(labels[row_index]),
        "gross_return": float(gross[row_index]),
        "probability_up": float(probability),
        "long_signal": bool(probability >= 0.5),
        "fear_greed_value": int(fng_values[row_index]),
        "sentiment_source_available_at": fng_available[row_index],
        "sentiment_source_observation_time_utc": fng_observations[row_index],
        "sentiment_source_age_seconds": int(fng_ages[row_index]),
    }


def _fit_arm_predictions(
    arm: str,
    horizon: int,
    plan: FoldPlan,
    enough: bool,
    train: Sequence[int],
    test: Sequence[int],
    y_train: NDArray[Any],
    features: NDArray[Any],
    candles: Sequence[Candle],
    labels: NDArray[Any],
    gross: NDArray[Any],
    decision_times: Sequence[str],
    fng_values: NDArray[Any],
    fng_available: Sequence[str | None],
    fng_observations: Sequence[str | None],
    fng_ages: NDArray[Any],
    config: dict[str, Any],
) -> tuple[list[bool], list[dict[str, Any]]]:
    if not enough:
        return [False] * len(test), []
    matrix = features[:, :8] if arm == "control" else features[:, :9]
    model = build_model(config)
    model.fit(matrix[list(train)], y_train)
    probabilities = model.predict_proba(matrix[list(test)])[:, 1]
    trial_id = _trial_id_for_arm(arm, horizon)
    prediction_rows = [
        _prediction_row(
            trial_id,
            arm,
            horizon,
            plan.fold_id,
            row_index,
            float(probabilities[offset]),
            candles,
            labels,
            gross,
            decision_times,
            fng_values,
            fng_available,
            fng_observations,
            fng_ages,
        )
        for offset, row_index in enumerate(test)
    ]
    signals = [bool(value >= 0.5) for value in probabilities]
    return signals, prediction_rows


def _arm_portfolios(
    candles: Sequence[Candle],
    test: Sequence[int],
    signals: Sequence[bool],
    horizon: int,
    marks: Sequence[int],
    config: dict[str, Any],
) -> dict[str, PortfolioOutcome]:
    outcomes: dict[str, PortfolioOutcome] = {}
    for cost_name in COST_NAMES:
        fraction = float(
            config["portfolio_evaluation"]["cost_scenarios"][cost_name]["fraction"]
        )
        outcomes[cost_name] = simulate_portfolio(
            candles, test, signals, horizon, fraction, marks
        )
    return outcomes


def _baseline_portfolios(
    candles: Sequence[Candle],
    plan: FoldPlan,
    test: Sequence[int],
    majority: bool,
    horizon: int,
    config: dict[str, Any],
) -> dict[str, dict[str, PortfolioOutcome]]:
    baselines: dict[str, dict[str, PortfolioOutcome]] = {
        "always_long_same_horizon_same_cost": {},
        "training_fold_majority_direction": {},
        "no_trade_cash_baseline": {},
    }
    for cost_name in COST_NAMES:
        fraction = float(
            config["portfolio_evaluation"]["cost_scenarios"][cost_name]["fraction"]
        )
        baselines["always_long_same_horizon_same_cost"][cost_name] = simulate_portfolio(
            candles, test, [True] * len(test), horizon, fraction, plan.mark_indices
        )
        baselines["training_fold_majority_direction"][cost_name] = simulate_portfolio(
            candles, test, [majority] * len(test), horizon, fraction, plan.mark_indices
        )
        baselines["no_trade_cash_baseline"][cost_name] = _empty_metrics(
            candles, plan, horizon, fraction
        )
    return baselines


def _fold_summary(
    plan: FoldPlan,
    decision_times: Sequence[str],
    train: Sequence[int],
    test: Sequence[int],
    counts: Counter[int],
    enough: bool,
    majority: bool,
) -> dict[str, Any]:
    return {
        "fold_id": plan.fold_id,
        "first_test_decision_time_utc": decision_times[plan.first_test_index],
        "last_test_decision_time_utc": decision_times[plan.last_test_index],
        "train_rows_after_purge": len(train),
        "train_class_counts": {
            "class_0": counts.get(0, 0),
            "class_1": counts.get(1, 0),
        },
        "purged_train_rows": plan.purged_train_rows,
        "test_rows_after_horizon_rule": len(test),
        "excluded_test_rows": plan.excluded_test_rows,
        "training_preflight": "PASS" if enough else "INSUFFICIENT_SAMPLE",
        "fit_status": "EVALUATED" if enough else "INSUFFICIENT_SAMPLE",
        "majority_direction": "LONG" if majority else "CASH",
    }


def _evaluate_fold(
    candles: Sequence[Candle],
    features: NDArray[Any],
    labels: NDArray[Any],
    gross: NDArray[Any],
    plan: FoldPlan,
    horizon: int,
    decision_times: Sequence[str],
    fng_values: NDArray[Any],
    fng_available: Sequence[str | None],
    fng_observations: Sequence[str | None],
    fng_ages: NDArray[Any],
    config: dict[str, Any],
) -> FoldOutput:
    train, test, y_train, counts, enough, majority = _fold_training_state(
        labels, plan, config
    )
    arm_outcomes: dict[str, dict[str, PortfolioOutcome]] = {}
    predictions: list[dict[str, Any]] = []
    for arm in ARMS:
        signals, arm_predictions = _fit_arm_predictions(
            arm,
            horizon,
            plan,
            enough,
            train,
            test,
            y_train,
            features,
            candles,
            labels,
            gross,
            decision_times,
            fng_values,
            fng_available,
            fng_observations,
            fng_ages,
            config,
        )
        predictions.extend(arm_predictions)
        arm_outcomes[arm] = _arm_portfolios(
            candles, test, signals, horizon, plan.mark_indices, config
        )
    baseline_outcomes = _baseline_portfolios(
        candles, plan, test, majority, horizon, config
    )
    summary = _fold_summary(plan, decision_times, train, test, counts, enough, majority)
    return FoldOutput(summary, arm_outcomes, baseline_outcomes, tuple(predictions))


def _source_age_summary(
    ages: NDArray[Any], observations: Sequence[str | None]
) -> dict[str, Any]:
    valid = ages[np.isfinite(ages)]
    unique = {item for item in observations if item is not None}
    if valid.size == 0:
        return {"matched_decision_rows": 0, "unique_daily_observations_used": 0}
    return {
        "matched_decision_rows": int(valid.size),
        "unique_daily_observations_used": len(unique),
        "min_seconds": float(np.min(valid)),
        "median_seconds": float(np.median(valid)),
        "p95_seconds": float(np.percentile(valid, 95)),
        "max_seconds": float(np.max(valid)),
    }


def _build_trial(
    arm: str, horizon: int, fold_outputs: Sequence[FoldOutput], config: dict[str, Any]
) -> dict[str, Any]:
    trial_id = (
        f"NOBITEX-M5-2-{horizon}h-OHLCV-LR-BASELINE-01"
        if arm == "control"
        else f"NOBITEX-M5-2-{horizon}h-SENTIMENT-LR-AUGMENTED-01"
    )
    result: dict[str, Any] = {
        "trial_id": trial_id,
        "arm": arm,
        "horizon_hours": horizon,
        "feature_names": list(
            CONTROL_FEATURES if arm == "control" else TREATMENT_FEATURES
        ),
        "folds": [],
        "cost_scenarios": {},
    }
    for fold in fold_outputs:
        cost_detail: dict[str, Any] = {}
        for cost_name in COST_NAMES:
            outcome = fold.arm_outcomes[arm][cost_name]
            cost_detail[cost_name] = {
                "portfolio": outcome.metrics,
                "valid_fold": is_valid_fold(outcome),
                "stable_fold": is_stable_fold(outcome),
            }
        result["folds"].append({**fold.summary, "cost_scenarios": cost_detail})
    for cost_name in COST_NAMES:
        outcomes = [fold.arm_outcomes[arm][cost_name] for fold in fold_outputs]
        result["cost_scenarios"][cost_name] = cost_summary(outcomes, config)
    return result


def _paired_comparisons(trials: dict[str, Any]) -> dict[str, Any]:
    paired: dict[str, Any] = {}
    for horizon in HORIZONS:
        control_id = f"NOBITEX-M5-2-{horizon}h-OHLCV-LR-BASELINE-01"
        sentiment_id = f"NOBITEX-M5-2-{horizon}h-SENTIMENT-LR-AUGMENTED-01"
        paired[str(horizon)] = {}
        for cost_name in COST_NAMES:
            control = trials[control_id]["cost_scenarios"][cost_name]["pooled_oos"]
            sentiment = trials[sentiment_id]["cost_scenarios"][cost_name]["pooled_oos"]
            c_pf = control["net_profit_factor"]
            s_pf = sentiment["net_profit_factor"]
            higher_pf = (
                c_pf is not None and s_pf is not None and float(s_pf) > float(c_pf)
            )
            higher_pnl = float(
                sentiment["total_net_pnl_fraction_of_initial_equity"]
            ) > float(control["total_net_pnl_fraction_of_initial_equity"])
            paired[str(horizon)][cost_name] = {
                "control_net_profit_factor": c_pf,
                "sentiment_net_profit_factor": s_pf,
                "control_net_pnl_fraction": control[
                    "total_net_pnl_fraction_of_initial_equity"
                ],
                "sentiment_net_pnl_fraction": sentiment[
                    "total_net_pnl_fraction_of_initial_equity"
                ],
                "sentiment_netpf_strictly_higher": higher_pf,
                "sentiment_net_pnl_strictly_higher": higher_pnl,
                "incremental_value_passed": higher_pf and higher_pnl,
            }
    return paired


def _overall_verdict(trials: dict[str, Any], paired: dict[str, Any]) -> str:
    insufficient = False
    failed = False
    for horizon in HORIZONS:
        trial_id = f"NOBITEX-M5-2-{horizon}h-SENTIMENT-LR-AUGMENTED-01"
        for cost_name in COST_NAMES:
            row = trials[trial_id]["cost_scenarios"][cost_name]
            insufficient = (
                insufficient
                or row["verdict_before_dsr"] == "INSUFFICIENT_SAMPLE / NO-GO"
            )
            failed = failed or not row["absolute_gates_passed"]
            failed = (
                failed
                or not paired[str(horizon)][cost_name]["incremental_value_passed"]
            )
    if insufficient:
        return "INSUFFICIENT_SAMPLE / NO-GO"
    if failed:
        return "NO-GO"
    return "INSUFFICIENT_EVIDENCE"


def _baseline_report(folds: Sequence[FoldOutput]) -> dict[str, Any]:
    names = (
        "always_long_same_horizon_same_cost",
        "training_fold_majority_direction",
        "no_trade_cash_baseline",
    )
    report: dict[str, Any] = {}
    for name in names:
        report[name] = {}
        for cost in COST_NAMES:
            report[name][cost] = {
                "pooled_oos": aggregate_outcomes(
                    [fold.baseline_outcomes[name][cost] for fold in folds]
                ),
                "fold_metrics": [
                    fold.baseline_outcomes[name][cost].metrics for fold in folds
                ],
            }
    return report


@dataclass(frozen=True)
class EvaluationInputs:
    config: dict[str, Any]
    prereg_sha: str
    candles: list[Candle]
    source_manifest: dict[str, Any]
    sentiment_records: list[Mapping[str, object]]
    sentiment_manifest: dict[str, Any]
    decision_times: list[str]
    sentiment_values: NDArray[Any]
    sentiment_available: list[str | None]
    sentiment_observations: list[str | None]
    sentiment_ages: NDArray[Any]
    price_features: NDArray[Any]
    features: NDArray[Any]
    eligible_indices: tuple[int, ...]


@dataclass(frozen=True)
class HorizonOutput:
    trials: dict[str, Any]
    baselines: dict[str, Any]
    predictions: tuple[dict[str, Any], ...]
    fold_plan_records: tuple[dict[str, Any], ...]


def _prepare_evaluation_inputs() -> EvaluationInputs:
    config, prereg_sha = load_preregistration()
    if sklearn.__version__ != "1.9.0":
        raise RuntimeError("m5_2_sklearn_version_mismatch")
    candles, source_manifest = load_verified_ohlcv()
    sentiment_records, sentiment_manifest = load_verified_sentiment(config)
    decision_times = [
        (row.start + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        for row in candles
    ]
    aligned = align_sentiment(sentiment_records, decision_times)
    sentiment, available, observations, ages = aligned
    price_features = build_price_features(candles)
    features = np.column_stack((price_features, sentiment))
    eligible = tuple(
        index for index in range(len(candles)) if np.isfinite(features[index]).all()
    )
    if not eligible:
        raise ValueError("m5_2_no_common_paired_samples")
    return EvaluationInputs(
        config=config,
        prereg_sha=prereg_sha,
        candles=candles,
        source_manifest=source_manifest,
        sentiment_records=sentiment_records,
        sentiment_manifest=sentiment_manifest,
        decision_times=decision_times,
        sentiment_values=sentiment,
        sentiment_available=available,
        sentiment_observations=observations,
        sentiment_ages=ages,
        price_features=price_features,
        features=features,
        eligible_indices=eligible,
    )


def _build_horizon_targets(
    candles: Sequence[Candle], horizon: int
) -> tuple[NDArray[Any], NDArray[Any], list[bool]]:
    returns = np.full(len(candles), np.nan, dtype=np.float64)
    labels = np.zeros(len(candles), dtype=np.int64)
    for index in range(len(candles) - horizon):
        gross = candles[index + horizon].close / candles[index + 1].open - 1.0
        returns[index] = gross
        labels[index] = int(gross > 0.0)
    valid_labels = np.isfinite(returns).tolist()
    return returns, labels, valid_labels


def _fold_plan_record(plan: FoldPlan) -> dict[str, Any]:
    return {
        "fold_id": plan.fold_id,
        "first_test_index": plan.first_test_index,
        "last_test_index": plan.last_test_index,
        "train_rows_after_purge": len(plan.train_indices),
        "test_rows_after_horizon_rule": len(plan.test_indices),
        "purged_train_rows": plan.purged_train_rows,
        "excluded_test_rows": plan.excluded_test_rows,
    }


def _evaluate_horizon(inputs: EvaluationInputs, horizon: int) -> HorizonOutput:
    returns, labels, valid_labels = _build_horizon_targets(inputs.candles, horizon)
    plans = build_fold_plans(inputs.eligible_indices, valid_labels, horizon)
    fold_outputs = [
        _evaluate_fold(
            inputs.candles,
            inputs.features,
            labels,
            returns,
            plan,
            horizon,
            inputs.decision_times,
            inputs.sentiment_values,
            inputs.sentiment_available,
            inputs.sentiment_observations,
            inputs.sentiment_ages,
            inputs.config,
        )
        for plan in plans
    ]
    trial_results = {
        trial["trial_id"]: trial
        for trial in (
            _build_trial(arm, horizon, fold_outputs, inputs.config) for arm in ARMS
        )
    }
    predictions = tuple(row for fold in fold_outputs for row in fold.predictions)
    return HorizonOutput(
        trials=trial_results,
        baselines=_baseline_report(fold_outputs),
        predictions=predictions,
        fold_plan_records=tuple(_fold_plan_record(plan) for plan in plans),
    )


def _build_dataset_binding(inputs: EvaluationInputs) -> dict[str, Any]:
    return {
        "market": "BTCUSDT",
        "venue": "Nobitex",
        "timeframe": "1h",
        "scope": "PRE_HACK_ONLY",
        "row_count": len(inputs.candles),
        "start_utc": inputs.source_manifest["start_utc"],
        "end_utc": inputs.source_manifest["end_utc"],
        "csv_sha256": OHLCV_SHA256,
        "source_manifest_sha256": OHLCV_MANIFEST_SHA256,
        "data_lock_sha256": sha256_bytes(DATA_LOCK_PATH.read_bytes()),
        "missing_intervals": 0,
        "confirmatory_holdout": False,
    }


def _build_sentiment_binding(inputs: EvaluationInputs) -> dict[str, Any]:
    manifest_path = SENTIMENT_DIR / str(
        inputs.config["sentiment_source"]["snapshot_manifest"]
    )
    return {
        "provider": "Alternative.me Fear & Greed",
        "raw_sha256": SENTIMENT_RAW_SHA256,
        "normalized_sha256": SENTIMENT_NORMALIZED_SHA256,
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "record_count": len(inputs.sentiment_records),
        "market_scope": inputs.sentiment_manifest["market_scope"],
        "availability_policy": inputs.sentiment_manifest["availability_policy"],
        "provider_supplies_historical_available_at": False,
        "alignment": "latest_available_at_lte_decision_time_forward_fill",
        "source_age_seconds": _source_age_summary(
            inputs.sentiment_ages, inputs.sentiment_observations
        ),
        "effective_frequency": "daily, hourly repetitions are dependent observations",
    }


def _build_paired_sample_report(inputs: EvaluationInputs) -> dict[str, Any]:
    warmup_excluded = sum(
        1 for row in inputs.price_features if not np.isfinite(row).all()
    )
    no_prior_sentiment = sum(
        1
        for index in range(len(inputs.candles))
        if np.isfinite(inputs.price_features[index]).all()
        and not math.isfinite(inputs.sentiment_values[index])
    )
    return {
        "raw_ohlcv_rows": len(inputs.candles),
        "ohlcv_warmup_rows_excluded": warmup_excluded,
        "rows_without_prior_available_sentiment_excluded_from_both_arms": no_prior_sentiment,
        "common_paired_rows": len(inputs.eligible_indices),
        "same_rows_labels_folds_costs_and_position_sizing": True,
        "decision": "completed candle close(t)",
        "entry": "open(t+1)",
        "exit": "close(t+h)",
    }


def _build_fold_protocol_report(
    inputs: EvaluationInputs, fold_records: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    return {
        "scheme": inputs.config["walk_forward_protocol"]["scheme"],
        "block_construction": inputs.config["walk_forward_protocol"][
            "block_construction"
        ],
        "folds": fold_records,
        "train_purge": "label exit close must be strictly before first test decision timestamp",
        "test_boundary": "label exit close must be inside the same test block",
    }


def _count_successful_fits(trials: dict[str, Any]) -> int:
    return sum(
        1
        for trial in trials.values()
        for fold in trial["folds"]
        if fold["fit_status"] == "EVALUATED"
    )


def _build_model_report(
    inputs: EvaluationInputs, trials: dict[str, Any]
) -> dict[str, Any]:
    config = inputs.config
    return {
        "estimator": config["model"]["estimator"],
        "pipeline": config["model"]["pipeline"],
        "parameters": config["model"]["parameters"],
        "threshold": 0.5,
        "sklearn_version": sklearn.__version__,
        "scaler_fit_on_training_fold_only": True,
        "fit_count": _count_successful_fits(trials),
    }


def _sentiment_absolute_gates_passed(trials: dict[str, Any]) -> bool:
    for horizon in HORIZONS:
        trial_id = f"NOBITEX-M5-2-{horizon}h-SENTIMENT-LR-AUGMENTED-01"
        for cost_name in COST_NAMES:
            result = trials[trial_id]["cost_scenarios"][cost_name]
            if not result["absolute_gates_passed"]:
                return False
    return True


def _paired_incremental_value_passed(paired: dict[str, Any]) -> bool:
    return all(
        paired[str(horizon)][cost_name]["incremental_value_passed"]
        for horizon in HORIZONS
        for cost_name in COST_NAMES
    )


def _build_economic_acceptance(
    trials: dict[str, Any], paired: dict[str, Any]
) -> dict[str, Any]:
    return {
        "overall_verdict": _overall_verdict(trials, paired),
        "sentiment_absolute_gates_passed": _sentiment_absolute_gates_passed(trials),
        "paired_incremental_value_passed": _paired_incremental_value_passed(paired),
        "first_economic_edge_claim_permitted": False,
    }


def _build_dsr_report() -> dict[str, str]:
    return {
        "status": "NOT_COMPUTABLE",
        "GLOBAL_N": "UNKNOWN",
        "GLOBAL_DSR": "NOT_COMPUTABLE",
        "reason": "Complete historical hypothesis-family accounting and full distribution inputs are unavailable.",
    }


def _build_safety_report() -> dict[str, Any]:
    return {
        "new_data_acquisition": False,
        "external_market_data_requests": 0,
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "paper_orders": 0,
        "holdout_bytes_read": False,
        "posthack_data_read": False,
        "local_execution": False,
        "exchange_network_access": False,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }


def _build_evaluation_report(
    inputs: EvaluationInputs,
    trials: dict[str, Any],
    paired: dict[str, Any],
    baselines: dict[str, Any],
    fold_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    config = inputs.config
    return {
        "schema_version": "alix.m5_2.evaluation-report.v1",
        "evaluation_status": "COMPLETED_REAL_PREHACK_RESEARCH",
        "phase": "M5.2",
        "preregistration_id": config["preregistration"]["id"],
        "preregistration_sha256": inputs.prereg_sha,
        "dataset_binding": _build_dataset_binding(inputs),
        "sentiment_binding": _build_sentiment_binding(inputs),
        "paired_sample": _build_paired_sample_report(inputs),
        "fold_protocol": _build_fold_protocol_report(inputs, fold_records),
        "model": _build_model_report(inputs, trials),
        "cost_scenarios": config["portfolio_evaluation"]["cost_scenarios"],
        "trial_results": trials,
        "paired_comparison": paired,
        "baselines_by_horizon": baselines,
        "dsr": _build_dsr_report(),
        "economic_acceptance": _build_economic_acceptance(trials, paired),
        "safety": _build_safety_report(),
    }


def run_evaluation() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inputs = _prepare_evaluation_inputs()
    trials: dict[str, Any] = {}
    baselines: dict[str, Any] = {}
    predictions: list[dict[str, Any]] = []
    fold_records: dict[str, list[dict[str, Any]]] = {}
    for horizon in HORIZONS:
        result = _evaluate_horizon(inputs, horizon)
        trials.update(result.trials)
        baselines[str(horizon)] = result.baselines
        predictions.extend(result.predictions)
        fold_records[str(horizon)] = list(result.fold_plan_records)
    paired = _paired_comparisons(trials)
    report = _build_evaluation_report(inputs, trials, paired, baselines, fold_records)
    return report, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    args = parser.parse_args()
    require_owner_authorization()
    report, predictions = run_evaluation()
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "trial_id",
        "horizon_hours",
        "arm",
        "fold_id",
        "decision_time_utc",
        "entry_time_utc",
        "exit_time_utc",
        "label",
        "gross_return",
        "probability_up",
        "long_signal",
        "fear_greed_value",
        "sentiment_source_available_at",
        "sentiment_source_observation_time_utc",
        "sentiment_source_age_seconds",
    ]
    with args.predictions_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    print(
        json.dumps(
            {
                "evaluation_status": report["evaluation_status"],
                "overall_verdict": report["economic_acceptance"]["overall_verdict"],
                "trial_count": len(report["trial_results"]),
                "prediction_rows": len(predictions),
                "preregistration_sha256": report["preregistration_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
