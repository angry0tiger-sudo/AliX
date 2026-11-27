"""Run the locked M6 three-class BTCUSDT Triple-Barrier evaluation in GitHub Actions."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)  # type: ignore[import-untyped]
import sklearn  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/m6/triple_barrier_3class_preregistration.json"
DATA_PATH = ROOT / "data/nobitex/prehack/nobitex-btcusdt-1h.csv"
MANIFEST_PATH = ROOT / "data/nobitex/prehack/nobitex-btcusdt-1h.manifest.json"
DATA_LOCK_PATH = ROOT / "data/nobitex/prehack/m5_2_btcusdt_1h_data_lock.json"

PREREG_SHA256 = "f94af9eb6f6003818beb88fcdbfbebe2ac2d7ac8a42281529db2781b44962e42"
OHLCV_SHA256 = "0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e"
MANIFEST_SHA256 = "b50fea25f30d282c2eafcb262751367e11d6490eda0d463de5926b660eaf1a26"
DATA_LOCK_SHA256 = "e66616ee3c13e6ce32f94577c4f46480390eea30c774ac6987e8d519d51d4acf"
FEATURE_NAMES = (
    "log_return_1",
    "log_return_4",
    "log_return_24",
    "realized_volatility_6",
    "realized_volatility_24",
    "range_fraction_1",
    "atr_fraction_14",
    "log_volume_change_1",
)
CLASS_NAMES = {0: "TP_FIRST", 1: "SL_FIRST", 2: "VERTICAL_TIMEOUT"}
OUTCOME_CLASSES = {"TP_FIRST": 0, "SL_FIRST": 1, "VERTICAL_TIMEOUT": 2}
EXPECTED_COLUMNS = ("date", "open", "high", "low", "close", "volume")
HORIZONS = {"1h": 1, "4h": 4}
COSTS = {"Cost-50": (50, 0.005), "Cost-90": (90, 0.009)}


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    base_start_index: int
    base_end_index: int


@dataclass(frozen=True)
class Event:
    outcome: str
    label: int | None
    entry_base_index: int
    max_end_base_index: int
    exit_base_index: int
    entry_price: float
    exit_price: float
    take_profit_price: float
    stop_loss_price: float


@dataclass(frozen=True)
class FoldPlan:
    fold_id: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    mark_indices: tuple[int, ...]
    first_test_entry_base_index: int
    last_test_base_index: int
    purged_train_rows: int
    excluded_test_rows: int


@dataclass(frozen=True)
class Trade:
    decision_index: int
    entry_base_index: int
    exit_base_index: int
    notional: float
    pnl: float
    outcome: str


@dataclass(frozen=True)
class PortfolioOutcome:
    metrics: dict[str, Any]
    nav: tuple[float, ...]
    pnl: float
    trades: tuple[Trade, ...]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, error_code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError(error_code) from None
    if not isinstance(value, dict):
        raise ValueError(error_code)
    return cast(dict[str, Any], value)


def load_preregistration() -> tuple[dict[str, Any], str]:
    raw = CONFIG_PATH.read_bytes()
    digest = sha256_bytes(raw)
    if digest != PREREG_SHA256:
        raise ValueError("m6_preregistration_sha256_mismatch")
    config = read_json(CONFIG_PATH, "m6_preregistration_invalid")
    registration = config["preregistration"]
    checks = (
        (registration["id"] == "ALIX-M6-BTCUSDT-TRIPLE-BARRIER-3CLASS-G1", "m6_preregistration_id_mismatch"),
        (registration["status"] == "PREREGISTERED_LOCKED", "m6_preregistration_not_locked"),
        (config["scope"]["market"] == "BTCUSDT" and config["scope"]["venue"] == "Nobitex", "m6_market_scope_mismatch"),
        (tuple(config["features"]["feature_names"]) == FEATURE_NAMES, "m6_feature_contract_mismatch"),
        (config["target"]["class_mapping"] == OUTCOME_CLASSES, "m6_class_mapping_mismatch"),
        (config["target"]["barriers"]["multiplier_k"] == 2.0, "m6_barrier_multiplier_mismatch"),
        (config["target"]["vertical_barrier"]["max_holding_hours"] == 24, "m6_vertical_horizon_mismatch"),
        (config["signal"]["tp_first_probability_threshold"] == 0.60, "m6_signal_threshold_mismatch"),
        (config["walk_forward"]["fold_count"] == 4, "m6_fold_count_mismatch"),
        (config["acceptance_gates"]["net_profit_factor_min"] == 1.10, "m6_netpf_gate_mismatch"),
        (config["acceptance_gates"]["completed_oos_trades_min"] == 100, "m6_trade_count_gate_mismatch"),
        (config["acceptance_gates"]["max_drawdown_fraction_max"] == 0.15, "m6_maxdd_gate_mismatch"),
    )
    for valid, error_code in checks:
        if not valid:
            raise ValueError(error_code)
    return config, digest


def require_owner_authorization() -> None:
    required = {
        "ALIX_M6_OWNER_AUTHORIZATION": "true",
        "ALIX_DRY_RUN": "true",
        "ALIX_ALLOW_LIVE_TRADING": "false",
        "GITHUB_ACTIONS": "true",
    }
    for name, expected in required.items():
        if os.environ.get(name) != expected:
            raise PermissionError("m6_execution_guard_failed:" + name)


def _parse_datetime(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("m6_timestamp_missing_timezone")
    return parsed.astimezone(timezone.utc)


def _parse_candle(row: dict[str, str], index: int) -> Candle:
    timestamp = _parse_datetime(row["date"])
    values = [float(row[name]) for name in ("open", "high", "low", "close", "volume")]
    opened, high, low, close, volume = values
    if not all(math.isfinite(item) for item in values):
        raise ValueError("m6_ohlcv_non_finite")
    if min(opened, high, low, close) <= 0.0 or volume < 0.0:
        raise ValueError("m6_ohlcv_out_of_range")
    if high < max(opened, low, close) or low > min(opened, high, close):
        raise ValueError("m6_ohlc_invariant_failed")
    return Candle(timestamp, opened, high, low, close, volume, index, index)


def _validate_data_bindings() -> dict[str, Any]:
    if sha256_bytes(DATA_PATH.read_bytes()) != OHLCV_SHA256:
        raise ValueError("m6_ohlcv_sha256_mismatch")
    if sha256_bytes(MANIFEST_PATH.read_bytes()) != MANIFEST_SHA256:
        raise ValueError("m6_manifest_sha256_mismatch")
    if sha256_bytes(DATA_LOCK_PATH.read_bytes()) != DATA_LOCK_SHA256:
        raise ValueError("m6_data_lock_sha256_mismatch")
    manifest = read_json(MANIFEST_PATH, "m6_manifest_invalid")
    lock = read_json(DATA_LOCK_PATH, "m6_data_lock_invalid")
    checks = (
        manifest["dataset_window"] == "PRE_HACK_NOBITEX_ONLY",
        manifest["symbol"] == "BTCUSDT",
        manifest["pair"] == "BTC/USDT",
        manifest["interval"] == "1h",
        manifest["row_count"] == 31801,
        manifest["missing_intervals"] == 0,
        manifest["csv_sha256"] == OHLCV_SHA256,
        manifest["public_api_only"] is True,
        manifest["credentials_used"] is False,
        manifest["private_api_calls"] == 0,
        manifest["real_order_attempts"] == 0,
        lock["venue"] == "Nobitex",
        lock["market"] == "BTCUSDT",
        lock["market_type"] == "Spot",
        lock["timeframe"] == "1h",
        lock["data_scope"] == "PRE_HACK_ONLY",
        lock["row_count"] == 31801,
        lock["raw_sha256"] == OHLCV_SHA256,
        lock["source_manifest_sha256"] == MANIFEST_SHA256,
        lock["gap_report"]["missing_intervals"] == 0,
        lock["gap_report"]["off_grid_intervals"] == 0,
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
    if not all(checks):
        raise ValueError("m6_data_scope_or_safety_binding_invalid")
    return manifest


def _read_hourly_csv() -> list[Candle]:
    candles: list[Candle] = []
    try:
        with DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
                raise ValueError("m6_ohlcv_columns_invalid")
            for index, row in enumerate(reader):
                candles.append(_parse_candle(cast(dict[str, str], row), index))
    except (OSError, csv.Error, KeyError, TypeError, ValueError) as exc:
        raise ValueError("m6_ohlcv_csv_invalid:" + type(exc).__name__) from None
    if len(candles) != 31801:
        raise ValueError("m6_ohlcv_row_count_invalid")
    return candles


def _validate_hourly_timeline(
    candles: Sequence[Candle], manifest: dict[str, Any]
) -> None:
    for previous, current in zip(candles, candles[1:]):
        if (current.timestamp - previous.timestamp).total_seconds() != 3600:
            raise ValueError("m6_ohlcv_gap_or_grid_violation")
    first = candles[0].timestamp.isoformat().replace("+00:00", "Z")
    last = candles[-1].timestamp.isoformat().replace("+00:00", "Z")
    if first != manifest["start_utc"] or last != manifest["end_utc"]:
        raise ValueError("m6_ohlcv_window_mismatch")


def load_verified_hourly_data() -> tuple[list[Candle], dict[str, Any]]:
    manifest = _validate_data_bindings()
    candles = _read_hourly_csv()
    _validate_hourly_timeline(candles, manifest)
    return candles, manifest


def aggregate_four_hour(hourly: Sequence[Candle]) -> list[Candle]:
    result: list[Candle] = []
    for start in range(0, len(hourly) - 3, 4):
        group = hourly[start : start + 4]
        if len(group) != 4:
            continue
        if any(
            (group[index].timestamp - group[index - 1].timestamp).total_seconds() != 3600
            for index in range(1, 4)
        ):
            raise ValueError("m6_four_hour_group_not_contiguous")
        result.append(
            Candle(
                timestamp=group[0].timestamp,
                open=group[0].open,
                high=max(item.high for item in group),
                low=min(item.low for item in group),
                close=group[-1].close,
                volume=sum(item.volume for item in group),
                base_start_index=group[0].base_start_index,
                base_end_index=group[-1].base_end_index,
            )
        )
    if not result:
        raise ValueError("m6_no_complete_four_hour_bars")
    return result


def build_features(candles: Sequence[Candle]) -> tuple[NDArray[Any], NDArray[Any]]:
    count = len(candles)
    close = np.asarray([item.close for item in candles], dtype=np.float64)
    high = np.asarray([item.high for item in candles], dtype=np.float64)
    low = np.asarray([item.low for item in candles], dtype=np.float64)
    volume = np.asarray([item.volume for item in candles], dtype=np.float64)
    log_close = np.log(close)
    returns = np.full(count, np.nan, dtype=np.float64)
    returns[1:] = np.diff(log_close)
    true_range = high - low
    true_range[1:] = np.maximum.reduce(
        (
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        )
    )
    atr = np.full(count, np.nan, dtype=np.float64)
    if count >= 14:
        atr[13] = float(np.mean(true_range[:14]))
        for index in range(14, count):
            atr[index] = (13.0 * atr[index - 1] + true_range[index]) / 14.0
    features = np.full((count, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    features[:, 0] = returns
    features[4:, 1] = log_close[4:] - log_close[:-4]
    features[24:, 2] = log_close[24:] - log_close[:-24]
    for index in range(5, count):
        features[index, 3] = float(np.std(returns[index - 5 : index + 1], ddof=0))
    for index in range(23, count):
        features[index, 4] = float(np.std(returns[index - 23 : index + 1], ddof=0))
    features[:, 5] = (high - low) / close
    atr_fraction = atr / close
    features[:, 6] = atr_fraction
    volume_change = np.full(count, np.nan, dtype=np.float64)
    volume_change[1:] = np.diff(np.log1p(volume))
    features[:, 7] = volume_change
    return features, atr_fraction


def _opening_gap_event(
    candle: Candle,
    offset: int,
    entry_base_index: int,
    max_end: int,
    entry_price: float,
    take_profit: float,
    stop_loss: float,
) -> Event | None:
    if offset == 0:
        return None
    if candle.open >= take_profit:
        return Event("TP_FIRST", 0, entry_base_index, max_end, candle.base_start_index, entry_price, candle.open, take_profit, stop_loss)
    if candle.open <= stop_loss:
        return Event("SL_FIRST", 1, entry_base_index, max_end, candle.base_start_index, entry_price, candle.open, take_profit, stop_loss)
    return None


def _intrabar_event(
    candle: Candle,
    entry_base_index: int,
    max_end: int,
    entry_price: float,
    take_profit: float,
    stop_loss: float,
) -> Event | None:
    hit_tp = candle.high >= take_profit
    hit_sl = candle.low <= stop_loss
    if hit_tp and hit_sl:
        return Event("AMBIGUOUS", None, entry_base_index, max_end, candle.base_start_index, entry_price, stop_loss, take_profit, stop_loss)
    if hit_tp:
        return Event("TP_FIRST", 0, entry_base_index, max_end, candle.base_start_index, entry_price, take_profit, take_profit, stop_loss)
    if hit_sl:
        return Event("SL_FIRST", 1, entry_base_index, max_end, candle.base_start_index, entry_price, stop_loss, take_profit, stop_loss)
    return None


def _resolve_event(
    hourly: Sequence[Candle], entry_base_index: int, atr_fraction: float
) -> Event | None:
    max_end = entry_base_index + 23
    if entry_base_index < 0 or max_end >= len(hourly):
        return None
    entry_price = hourly[entry_base_index].open
    distance = 2.0 * atr_fraction
    take_profit = entry_price * (1.0 + distance)
    stop_loss = entry_price * (1.0 - distance)
    if not math.isfinite(stop_loss) or stop_loss <= 0.0:
        return None
    for offset in range(24):
        candle = hourly[entry_base_index + offset]
        gap_event = _opening_gap_event(
            candle, offset, entry_base_index, max_end, entry_price, take_profit, stop_loss
        )
        if gap_event is not None:
            return gap_event
        touch_event = _intrabar_event(
            candle, entry_base_index, max_end, entry_price, take_profit, stop_loss
        )
        if touch_event is not None:
            return touch_event
    return Event(
        "VERTICAL_TIMEOUT", 2, entry_base_index, max_end, max_end,
        entry_price, hourly[max_end].close, take_profit, stop_loss
    )


def build_events(
    hourly: Sequence[Candle],
    decision_bars: Sequence[Candle],
    atr_fraction: Sequence[float],
) -> list[Event | None]:
    events: list[Event | None] = [None] * len(decision_bars)
    for index in range(len(decision_bars) - 1):
        value = float(atr_fraction[index])
        if math.isfinite(value) and value > 0.0:
            entry_index = decision_bars[index + 1].base_start_index
            events[index] = _resolve_event(hourly, entry_index, value)
    return events


def _training_fold_rows(
    candidates: Sequence[int], events: Sequence[Event | None], first_entry: int, embargo_hours: int
) -> tuple[tuple[int, ...], int]:
    rows: list[int] = []
    purged = 0
    for index in candidates:
        event = events[index]
        if event is None or event.label is None:
            continue
        if event.max_end_base_index < first_entry - embargo_hours:
            rows.append(index)
        else:
            purged += 1
    return tuple(rows), purged


def _test_fold_rows(
    candidates: Sequence[int], events: Sequence[Event | None], last_test_base: int
) -> tuple[int, ...]:
    return tuple(
        index for index in candidates
        if events[index] is not None and events[index].max_end_base_index <= last_test_base
    )


def build_fold_plans(
    eligible_indices: Sequence[int],
    decision_bars: Sequence[Candle],
    events: Sequence[Event | None],
    embargo_hours: int = 24,
) -> tuple[FoldPlan, ...]:
    if len(eligible_indices) < 5:
        raise ValueError("m6_insufficient_feature_rows_for_walk_forward")
    if any(left >= right for left, right in zip(eligible_indices, eligible_indices[1:])):
        raise ValueError("m6_eligible_indices_not_strictly_ordered")
    edges = [index * len(eligible_indices) // 5 for index in range(6)]
    plans: list[FoldPlan] = []
    for fold_index in range(4):
        train_candidates = eligible_indices[: edges[fold_index + 1]]
        test_candidates = eligible_indices[edges[fold_index + 1] : edges[fold_index + 2]]
        if not test_candidates:
            raise ValueError("m6_empty_test_block")
        first_test = test_candidates[0]
        last_test = test_candidates[-1]
        first_entry = decision_bars[first_test + 1].base_start_index
        last_test_base = decision_bars[last_test].base_end_index
        train_rows, purged = _training_fold_rows(
            train_candidates, events, first_entry, embargo_hours
        )
        test_rows = _test_fold_rows(test_candidates, events, last_test_base)
        marks = tuple(
            range(decision_bars[first_test].base_end_index, last_test_base + 1)
        )
        plans.append(
            FoldPlan(
                fold_id=f"WF-{fold_index + 1:02d}",
                train_indices=train_rows,
                test_indices=test_rows,
                mark_indices=marks,
                first_test_entry_base_index=first_entry,
                last_test_base_index=last_test_base,
                purged_train_rows=purged,
                excluded_test_rows=len(test_candidates) - len(test_rows),
            )
        )
    return tuple(plans)


def build_model(config: dict[str, Any]) -> RandomForestClassifier:
    params = config["model"]["parameters"]
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        criterion=str(params["criterion"]),
        max_depth=int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=float(params["max_features"]),
        bootstrap=bool(params["bootstrap"]),
        max_samples=float(params["max_samples"]),
        class_weight=str(params["class_weight"]),
        random_state=int(params["random_state"]),
        n_jobs=int(params["n_jobs"]),
    )


def is_long_signal(probability_tp_first: float, threshold: float = 0.60) -> bool:
    return probability_tp_first > threshold


def net_profit_factor(trades: Sequence[Trade]) -> float | None:
    wins = sum(item.pnl for item in trades if item.pnl > 0.0)
    losses = -sum(item.pnl for item in trades if item.pnl < 0.0)
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


def _trade_list(
    test_indices: Sequence[int],
    events: Sequence[Event | None],
    probabilities: Sequence[float],
    notional: float,
    cost: float,
) -> tuple[Trade, ...]:
    trades: list[Trade] = []
    for decision, probability in zip(test_indices, probabilities):
        if not is_long_signal(float(probability)):
            continue
        event = events[decision]
        if event is None:
            raise ValueError("m6_signal_without_completed_event")
        gross = event.exit_price / event.entry_price - 1.0
        pnl = notional * (gross - cost)
        trades.append(
            Trade(
                decision_index=decision,
                entry_base_index=event.entry_base_index,
                exit_base_index=event.exit_base_index,
                notional=notional,
                pnl=pnl,
                outcome=event.outcome,
            )
        )
    return tuple(sorted(trades, key=lambda item: (item.entry_base_index, item.exit_base_index)))


def _mark_portfolio(
    hourly: Sequence[Candle], trades: Sequence[Trade], mark_indices: Sequence[int], cost: float
) -> tuple[list[float], int]:
    nav = [1.0]
    max_concurrent = 0
    for mark in mark_indices:
        realized = sum(item.pnl for item in trades if item.exit_base_index <= mark)
        unrealized = sum(
            item.notional
            * (hourly[mark].close / hourly[item.entry_base_index].open - 1.0 - cost)
            for item in trades
            if item.entry_base_index <= mark < item.exit_base_index
        )
        active_count = sum(
            1 for item in trades if item.entry_base_index <= mark <= item.exit_base_index
        )
        max_concurrent = max(max_concurrent, active_count)
        nav.append(1.0 + realized + unrealized)
    return nav, max_concurrent


def simulate_portfolio(
    hourly: Sequence[Candle],
    test_indices: Sequence[int],
    events: Sequence[Event | None],
    probabilities: Sequence[float],
    decision_interval_hours: int,
    cost: float,
    mark_indices: Sequence[int],
) -> PortfolioOutcome:
    if len(test_indices) != len(probabilities):
        raise ValueError("m6_probability_count_mismatch")
    if not mark_indices:
        return _empty_portfolio()
    slots = 24 // decision_interval_hours + 1
    notional = 1.0 / float(slots)
    trades = _trade_list(test_indices, events, probabilities, notional, cost)
    nav, max_concurrent = _mark_portfolio(hourly, trades, mark_indices, cost)
    if max_concurrent > slots:
        raise ValueError("m6_position_concurrency_cap_breached")
    pnl = float(sum(item.pnl for item in trades))
    metrics = {
        "completed_oos_trades": len(trades),
        "net_profit_factor": net_profit_factor(trades),
        "total_net_pnl_fraction_of_initial_equity": pnl,
        "max_drawdown": max_drawdown(nav),
        "maximum_concurrent_positions": max_concurrent,
        "position_notional_fraction": notional,
        "ambiguous_trades_conservatively_charged_as_sl": sum(
            1 for item in trades if item.outcome == "AMBIGUOUS"
        ),
        "turnover_fraction_of_initial_equity": float(len(trades) * notional),
    }
    return PortfolioOutcome(metrics, tuple(nav), pnl, trades)


def _empty_portfolio(mark_count: int = 0) -> PortfolioOutcome:
    return PortfolioOutcome(
        {
            "completed_oos_trades": 0,
            "net_profit_factor": None,
            "total_net_pnl_fraction_of_initial_equity": 0.0,
            "max_drawdown": 0.0,
            "maximum_concurrent_positions": 0,
            "position_notional_fraction": 0.0,
            "ambiguous_trades_conservatively_charged_as_sl": 0,
            "turnover_fraction_of_initial_equity": 0.0,
        },
        tuple([1.0] * (mark_count + 1)),
        0.0,
        (),
    )


def aggregate_portfolios(outcomes: Sequence[PortfolioOutcome]) -> dict[str, Any]:
    trades = tuple(item for outcome in outcomes for item in outcome.trades)
    nav = [1.0]
    carried_pnl = 0.0
    for outcome in outcomes:
        nav.extend(1.0 + carried_pnl + (value - 1.0) for value in outcome.nav[1:])
        carried_pnl += outcome.pnl
    return {
        "completed_oos_trades": len(trades),
        "net_profit_factor": net_profit_factor(trades),
        "total_net_pnl_fraction_of_initial_equity": float(sum(item.pnl for item in trades)),
        "max_drawdown": max_drawdown(nav),
        "maximum_concurrent_positions": max(
            (int(item.metrics["maximum_concurrent_positions"]) for item in outcomes), default=0
        ),
        "ambiguous_trades_conservatively_charged_as_sl": sum(
            int(item.metrics["ambiguous_trades_conservatively_charged_as_sl"])
            for item in outcomes
        ),
    }


def is_valid_fold(outcome: PortfolioOutcome) -> bool:
    return int(outcome.metrics["completed_oos_trades"]) >= 20


def is_stable_fold(outcome: PortfolioOutcome) -> bool:
    pf = outcome.metrics["net_profit_factor"]
    return is_valid_fold(outcome) and outcome.pnl > 0.0 and pf is not None and float(pf) > 1.0


def check_gates(
    pooled: dict[str, Any], valid_folds: int, stable_folds: int, config: dict[str, Any]
) -> dict[str, bool]:
    gates = config["acceptance_gates"]
    pf = pooled["net_profit_factor"]
    return {
        "netpf_gte_1_10": pf is not None and float(pf) >= float(gates["net_profit_factor_min"]),
        "trades_gte_100": int(pooled["completed_oos_trades"]) >= int(gates["completed_oos_trades_min"]),
        "maxdd_lte_15_percent": float(pooled["max_drawdown"]) <= float(gates["max_drawdown_fraction_max"]),
        "at_least_3_valid_folds": valid_folds >= int(gates["minimum_valid_folds"]),
        "at_least_3_stable_folds": stable_folds >= int(gates["minimum_stable_folds"]),
    }


def _classification_metrics(
    events: Sequence[Event | None], indices: Sequence[int], predictions: Sequence[int]
) -> dict[str, Any]:
    actual: list[int] = []
    predicted: list[int] = []
    for index, prediction in zip(indices, predictions):
        event = events[index]
        if event is not None and event.label is not None:
            actual.append(event.label)
            predicted.append(int(prediction))
    if not actual:
        return {
            "scored_rows": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "confusion_matrix": [],
        }
    return {
        "scored_rows": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "confusion_matrix_labels_tp_sl_timeout_0_1_2": confusion_matrix(
            actual, predicted, labels=[0, 1, 2]
        ).tolist(),
    }


def _fold_record_without_fit(
    plan: FoldPlan, class_counts: dict[str, int], reason: str
) -> dict[str, Any]:
    return {
        "fold_id": plan.fold_id,
        "status": "INSUFFICIENT_SAMPLE",
        "reason": reason,
        "train_rows": len(plan.train_indices),
        "train_class_counts": class_counts,
        "test_rows": len(plan.test_indices),
        "purged_train_rows": plan.purged_train_rows,
        "excluded_test_rows": plan.excluded_test_rows,
    }


def _training_labels_and_counts(
    plan: FoldPlan, events: Sequence[Event | None]
) -> tuple[list[int], Counter[str]]:
    labels: list[int] = []
    counts: Counter[str] = Counter()
    for index in plan.train_indices:
        event = events[index]
        if event is not None and event.label is not None:
            labels.append(event.label)
            counts[CLASS_NAMES[event.label]] += 1
    return labels, counts


def _training_failure_reason(
    plan: FoldPlan, counts: Counter[str], config: dict[str, Any]
) -> str | None:
    if not plan.test_indices:
        return "no_complete_test_events"
    rules = config["walk_forward"]["training_rules"]
    minimum_rows = int(rules["minimum_training_rows"])
    minimum_per_class = int(rules["minimum_training_rows_per_class"])
    if len(plan.train_indices) < minimum_rows:
        return "minimum_training_rows"
    missing = [
        name for name in CLASS_NAMES.values()
        if counts.get(name, 0) < minimum_per_class
    ]
    if missing:
        return "minimum_training_class_count:" + ",".join(sorted(missing))
    return None


def _predict_fold(
    features: NDArray[Any],
    plan: FoldPlan,
    train_labels: Sequence[int],
    config: dict[str, Any],
) -> tuple[NDArray[Any], NDArray[Any], list[float], dict[int, int]]:
    x_train = np.asarray(features[list(plan.train_indices)], dtype=np.float64)
    x_test = np.asarray(features[list(plan.test_indices)], dtype=np.float64)
    model = build_model(config)
    model.fit(x_train, np.asarray(train_labels, dtype=np.int64))
    probabilities = model.predict_proba(x_test)
    predictions = model.predict(x_test)
    columns = {int(value): index for index, value in enumerate(model.classes_)}
    if set(columns) != {0, 1, 2}:
        raise ValueError("m6_fitted_model_missing_registered_class")
    tp_probabilities = [float(row[columns[0]]) for row in probabilities]
    return probabilities, predictions, tp_probabilities, columns


def _make_prediction_rows(
    hourly: Sequence[Candle],
    decision_bars: Sequence[Candle],
    events: Sequence[Event | None],
    plan: FoldPlan,
    horizon_name: str,
    probabilities: NDArray[Any],
    predictions: NDArray[Any],
    tp_probabilities: Sequence[float],
    columns: dict[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, decision in enumerate(plan.test_indices):
        event = events[decision]
        if event is None:
            raise ValueError("m6_test_event_missing_after_fold_filter")
        rows.append({
            "trial_id": f"NOBITEX-M6-{horizon_name}-RF-3CLASS-01",
            "horizon": horizon_name,
            "fold_id": plan.fold_id,
            "decision_time_utc": decision_bars[decision].timestamp.isoformat().replace("+00:00", "Z"),
            "entry_time_utc": hourly[event.entry_base_index].timestamp.isoformat().replace("+00:00", "Z"),
            "event_outcome": event.outcome,
            "true_class": CLASS_NAMES[event.label] if event.label is not None else None,
            "predicted_class": CLASS_NAMES[int(predictions[row_index])],
            "p_tp_first": float(tp_probabilities[row_index]),
            "p_sl_first": float(probabilities[row_index][columns[1]]),
            "p_vertical_timeout": float(probabilities[row_index][columns[2]]),
            "long_signal": is_long_signal(tp_probabilities[row_index]),
            "entry_price": float(event.entry_price),
            "exit_price_for_economic_ledger": float(event.exit_price),
            "exit_outcome_for_economic_ledger": (
                "SL_FIRST_CONSERVATIVE" if event.outcome == "AMBIGUOUS" else event.outcome
            ),
        })
    return rows


def _fold_cost_outcomes(
    hourly: Sequence[Candle],
    events: Sequence[Event | None],
    plan: FoldPlan,
    tp_probabilities: Sequence[float],
    horizon_name: str,
) -> dict[str, PortfolioOutcome]:
    outcomes: dict[str, PortfolioOutcome] = {}
    for cost_name, (_bps, fraction) in COSTS.items():
        outcomes[cost_name] = simulate_portfolio(
            hourly,
            plan.test_indices,
            events,
            tp_probabilities,
            HORIZONS[horizon_name],
            fraction,
            plan.mark_indices,
        )
    return outcomes


def _fold_test_counts(
    events: Sequence[Event | None], indices: Sequence[int]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for index in indices:
        event = events[index]
        if event is not None:
            counts[event.outcome] += 1
    return dict(counts)


def evaluate_fold(
    hourly: Sequence[Candle],
    decision_bars: Sequence[Candle],
    features: NDArray[Any],
    events: Sequence[Event | None],
    plan: FoldPlan,
    horizon_name: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, PortfolioOutcome], list[dict[str, Any]]]:
    train_labels, counts = _training_labels_and_counts(plan, events)
    failure_reason = _training_failure_reason(plan, counts, config)
    if failure_reason is not None:
        record = _fold_record_without_fit(plan, dict(counts), failure_reason)
        outcomes = {name: _empty_portfolio(len(plan.mark_indices)) for name in COSTS}
        return record, outcomes, []
    probabilities, predictions, tp_probabilities, columns = _predict_fold(
        features, plan, train_labels, config
    )
    prediction_rows = _make_prediction_rows(
        hourly, decision_bars, events, plan, horizon_name,
        probabilities, predictions, tp_probabilities, columns
    )
    class_metrics = _classification_metrics(
        events, plan.test_indices, [int(item) for item in predictions]
    )
    outcomes = _fold_cost_outcomes(
        hourly, events, plan, tp_probabilities, horizon_name
    )
    record = {
        "fold_id": plan.fold_id,
        "status": "FITTED",
        "train_rows": len(plan.train_indices),
        "train_class_counts": dict(counts),
        "test_rows": len(plan.test_indices),
        "test_event_counts": _fold_test_counts(events, plan.test_indices),
        "purged_train_rows": plan.purged_train_rows,
        "excluded_test_rows": plan.excluded_test_rows,
        "classification_metrics_excluding_ambiguous": class_metrics,
        "max_concurrent_capacity_slots": 24 // HORIZONS[horizon_name] + 1,
        "cost_scenarios": {
            name: {
                "metrics": outcome.metrics,
                "valid_fold": is_valid_fold(outcome),
                "stable_fold": is_stable_fold(outcome),
            }
            for name, outcome in outcomes.items()
        },
    }
    return record, outcomes, prediction_rows


def _summarize_cost(
    outcomes: Sequence[PortfolioOutcome], config: dict[str, Any]
) -> dict[str, Any]:
    pooled = aggregate_portfolios(outcomes)
    valid = sum(1 for item in outcomes if is_valid_fold(item))
    stable = sum(1 for item in outcomes if is_stable_fold(item))
    gates = check_gates(pooled, valid, stable, config)
    insufficient = pooled["completed_oos_trades"] < 100 or valid < 3
    verdict = "PASS" if all(gates.values()) else "INSUFFICIENT_SAMPLE / NO-GO" if insufficient else "NO-GO"
    return {
        "pooled_oos": pooled,
        "valid_folds": valid,
        "stable_folds": stable,
        "gates": gates,
        "verdict": verdict,
    }


def _eligible_decisions(
    decision_bars: Sequence[Candle],
    features: NDArray[Any],
    atr_fraction: NDArray[Any],
) -> list[int]:
    return [
        index
        for index in range(len(decision_bars) - 1)
        if np.isfinite(features[index]).all()
        and math.isfinite(float(atr_fraction[index]))
        and float(atr_fraction[index]) > 0.0
    ]


def _event_counts(
    eligible: Sequence[int], events: Sequence[Event | None]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for index in eligible:
        event = events[index]
        if event is not None:
            counts[event.outcome] += 1
    counts["UNAVAILABLE_OR_INCOMPLETE"] = sum(
        1 for index in eligible if events[index] is None
    )
    return dict(counts)


def evaluate_horizon(
    hourly: Sequence[Candle],
    timeframe: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decision_bars = list(hourly) if timeframe == "1h" else aggregate_four_hour(hourly)
    features, atr_fraction = build_features(decision_bars)
    events = build_events(hourly, decision_bars, atr_fraction)
    eligible = _eligible_decisions(decision_bars, features, atr_fraction)
    plans = build_fold_plans(
        eligible,
        decision_bars,
        events,
        embargo_hours=int(config["target"]["embargo_hours"]),
    )
    fold_records: list[dict[str, Any]] = []
    cost_outcomes: dict[str, list[PortfolioOutcome]] = {name: [] for name in COSTS}
    predictions: list[dict[str, Any]] = []
    for plan in plans:
        record, outcomes, rows = evaluate_fold(
            hourly, decision_bars, features, events, plan, timeframe, config
        )
        fold_records.append(record)
        predictions.extend(rows)
        for cost_name in COSTS:
            cost_outcomes[cost_name].append(outcomes[cost_name])
    cost_summaries = {
        name: _summarize_cost(outcomes, config)
        for name, outcomes in cost_outcomes.items()
    }
    return {
        "trial_id": f"NOBITEX-M6-{timeframe}-RF-3CLASS-01",
        "horizon": timeframe,
        "decision_bars": len(decision_bars),
        "feature_eligible_decisions": len(eligible),
        "label_counts": _event_counts(eligible, events),
        "folds": fold_records,
        "cost_scenarios": cost_summaries,
    }, predictions


def run_evaluation() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config, preregistration_sha = load_preregistration()
    hourly, manifest = load_verified_hourly_data()
    trial_results: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for timeframe in ("1h", "4h"):
        result, rows = evaluate_horizon(hourly, timeframe, config)
        trial_results[result["trial_id"]] = result
        prediction_rows.extend(rows)
    all_passed = all(
        summary["verdict"] == "PASS"
        for trial in trial_results.values()
        for summary in trial["cost_scenarios"].values()
    )
    any_insufficient = any(
        summary["verdict"] == "INSUFFICIENT_SAMPLE / NO-GO"
        for trial in trial_results.values()
        for summary in trial["cost_scenarios"].values()
    )
    overall = "ECONOMIC_GATES_PASS_EXPLORATORY_ONLY" if all_passed else "INSUFFICIENT_SAMPLE / NO-GO" if any_insufficient else "NO-GO"
    report = {
        "schema_version": "alix.m6.triple-barrier-3class-evaluation-report.v1",
        "evaluation_status": "COMPLETED_PREHACK_RESEARCH" if len(prediction_rows) > 0 else "COMPLETED_INSUFFICIENT_OOS",
        "commit_sha": os.environ.get("GITHUB_SHA", "UNKNOWN"),
        "preregistration_id": config["preregistration"]["id"],
        "preregistration_sha256": preregistration_sha,
        "model_runtime": {"scikit_learn": sklearn.__version__, "estimator": "RandomForestClassifier"},
        "dataset_binding": {
            "venue": "Nobitex",
            "market": "BTCUSDT",
            "timeframe": "1h",
            "rows": len(hourly),
            "start_utc": manifest["start_utc"],
            "end_utc": manifest["end_utc"],
            "csv_sha256": OHLCV_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "data_lock_sha256": DATA_LOCK_SHA256,
            "source_scope": "PRE_HACK_ONLY_HISTORICALLY_EXPOSED_EXPLORATORY",
        },
        "target": {
            "classes": {CLASS_NAMES[key]: key for key in sorted(CLASS_NAMES)},
            "tp_sl_multiplier": 2.0,
            "vertical_hours": 24,
            "ambiguous_policy": "excluded from classifier fit/metrics; charged as SL in OOS economics",
        },
        "feature_names": list(FEATURE_NAMES),
        "model_parameters": config["model"]["parameters"],
        "signal_rule": "P(TP_FIRST) > 0.60",
        "fold_protocol": config["walk_forward"],
        "trial_count": 2,
        "trial_results": trial_results,
        "economic_acceptance": {
            "overall_verdict": overall,
            "both_horizons_pass_both_costs": all_passed,
            "required_horizons": ["1h", "4h"],
            "required_costs": list(COSTS),
        },
        "dsr": {"status": "NOT_COMPUTABLE", "GLOBAL_N": "UNKNOWN"},
        "safety": {
            "data_scope": "PRE_HACK_ONLY",
            "new_data_acquisition": False,
            "onchain_data_read": False,
            "sentiment_data_read": False,
            "microstructure_data_read": False,
            "holdout_bytes_read": False,
            "posthack_data_read": False,
            "private_api_calls": 0,
            "exchange_network_access": False,
            "real_order_attempts": 0,
            "paper_orders": 0,
            "local_execution": False,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
        },
    }
    return report, prediction_rows


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
        "trial_id", "horizon", "fold_id", "decision_time_utc", "entry_time_utc",
        "event_outcome", "true_class", "predicted_class", "p_tp_first", "p_sl_first",
        "p_vertical_timeout", "long_signal", "entry_price", "exit_price_for_economic_ledger",
        "exit_outcome_for_economic_ledger",
    ]
    with args.predictions_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps({
        "evaluation_status": report["evaluation_status"],
        "overall_verdict": report["economic_acceptance"]["overall_verdict"],
        "trial_count": report["trial_count"],
        "prediction_rows": len(predictions),
        "preregistration_sha256": report["preregistration_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
