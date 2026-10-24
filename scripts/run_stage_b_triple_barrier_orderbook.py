"""Stage B Triple-Barrier and relative cross-venue Order Book research."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from scripts.run_stage_b_walk_forward import A_HOLDOUT, HOUR, SOURCE_RUN_ID, utc_text, verify_source_artifact
from services.meta_labeling.triple_barrier import TripleBarrierConfig, apply_triple_barrier

SCHEMA = "alix.stage-b.triple-barrier-orderbook.v1"
HORIZONS = (24, 48)
PT_MULTIPLIERS = (1.5, 2.0)
SL_MULTIPLIERS = (0.75, 1.0)
SLIPPAGE_BPS = (20.0, 35.0, 50.0)
FEE_BPS_SIDE = 25.0
PARTICIPATION_CAP = 0.01
BOOK_TIME = "snapshot_timestamp_utc"
BOOK_LEVELS = 20
BOOK_MAX_AGE_SECONDS = 60
INITIAL_TRAIN_ROWS = 3 * 365 * 24
OOS_WINDOW_ROWS = 90 * 24
PROBABILITY_THRESHOLD = 0.50
SAFETY: dict[str, object] = {
    "Champion": None,
    "order_permission": False,
    "trading_authorization": "NO-GO",
    "economic_acceptance": "NO-GO",
    "model_promoted": False,
    "production_model_updated": False,
    "strategy_registry_updated": False,
    "orders_or_execution_api_used": False,
    "private_api_calls": 0,
    "order_endpoint_calls": 0,
    "real_order_attempts": 0,
}


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
    fields = list(rows[0]) if rows else ["status", "reason"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validate_orderbook(book: DataFrame) -> DataFrame:
    required = {BOOK_TIME}
    for level in range(1, BOOK_LEVELS + 1):
        required.update(
            {
                f"bid_price_{level}",
                f"bid_qty_{level}",
                f"ask_price_{level}",
                f"ask_qty_{level}",
            }
        )
    missing = sorted(required - set(book.columns))
    if missing:
        raise ValueError(f"orderbook_schema_missing:{','.join(missing)}")
    if book.empty:
        raise ValueError("orderbook_dataset_empty")
    result = book.loc[:, sorted(required)].copy()
    result[BOOK_TIME] = pd.to_datetime(result[BOOK_TIME], utc=True, errors="raise")
    if result[BOOK_TIME].isna().any() or result[BOOK_TIME].duplicated().any():
        raise ValueError("orderbook_timestamp_invalid_or_duplicate")
    if not result[BOOK_TIME].is_monotonic_increasing:
        raise ValueError("orderbook_timestamps_not_monotonic")
    for level in range(1, BOOK_LEVELS + 1):
        for side in ("bid", "ask"):
            price_column = f"{side}_price_{level}"
            quantity_column = f"{side}_qty_{level}"
            result[price_column] = pd.to_numeric(result[price_column], errors="coerce")
            result[quantity_column] = pd.to_numeric(
                result[quantity_column], errors="coerce"
            )
            values = result[[price_column, quantity_column]].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"orderbook_non_finite:{side}:{level}")
            if (result[price_column] <= 0).any() or (
                result[quantity_column] < 0
            ).any():
                raise ValueError(f"orderbook_value_out_of_range:{side}:{level}")
    bid_prices = result[[f"bid_price_{i}" for i in range(1, 21)]].to_numpy(float)
    ask_prices = result[[f"ask_price_{i}" for i in range(1, 21)]].to_numpy(float)
    bid_quantities = result[[f"bid_qty_{i}" for i in range(1, 21)]].to_numpy(float)
    ask_quantities = result[[f"ask_qty_{i}" for i in range(1, 21)]].to_numpy(float)
    if np.any(np.diff(bid_prices, axis=1) > 0) or np.any(
        np.diff(ask_prices, axis=1) < 0
    ):
        raise ValueError("orderbook_prices_not_price_sorted")
    if np.any(bid_prices[:, 0] >= ask_prices[:, 0]):
        raise ValueError("orderbook_crossed_or_locked")
    if np.any(bid_quantities.sum(axis=1) <= 0) or np.any(
        ask_quantities.sum(axis=1) <= 0
    ):
        raise ValueError("orderbook_side_depth_empty")
    return result


def add_relative_orderbook_features(candles: DataFrame, book: DataFrame) -> DataFrame:
    """Join a recent prior snapshot and emit only normalized features."""
    left = candles.loc[:, ["date"]].copy()
    left["date"] = pd.to_datetime(left["date"], utc=True, errors="raise")
    if not (left["date"].diff().dropna() == pd.Timedelta(seconds=HOUR)).all():
        raise ValueError("orderbook_join_candles_not_hourly")
    left["decision_time"] = left["date"] + pd.Timedelta(seconds=HOUR)
    right = _validate_orderbook(book)
    bid_columns = [f"bid_qty_{i}" for i in range(1, BOOK_LEVELS + 1)]
    ask_columns = [f"ask_qty_{i}" for i in range(1, BOOK_LEVELS + 1)]
    bid5 = right[bid_columns[:5]].sum(axis=1)
    ask5 = right[ask_columns[:5]].sum(axis=1)
    bid20 = right[bid_columns].sum(axis=1)
    ask20 = right[ask_columns].sum(axis=1)
    right["imbalance_l5"] = (bid5 - ask5) / (bid5 + ask5)
    right["imbalance_l20"] = (bid20 - ask20) / (bid20 + ask20)
    midpoint = (right["bid_price_1"] + right["ask_price_1"]) / 2.0
    right["spread_bps"] = (
        (right["ask_price_1"] - right["bid_price_1"]) / midpoint * 10_000.0
    )
    keep = [BOOK_TIME, "imbalance_l5", "imbalance_l20", "spread_bps"]
    joined = pd.merge_asof(
        left.sort_values("decision_time"),
        right.loc[:, keep].sort_values(BOOK_TIME),
        left_on="decision_time",
        right_on=BOOK_TIME,
        direction="backward",
        tolerance=pd.Timedelta(seconds=BOOK_MAX_AGE_SECONDS),
        allow_exact_matches=True,
    )
    if joined[BOOK_TIME].isna().any():
        raise ValueError("orderbook_snapshot_missing_or_stale")
    if not np.isfinite(joined[keep[1:]].to_numpy(dtype=float)).all():
        raise ValueError("orderbook_relative_feature_non_finite")
    return joined.loc[:, ["date", *keep[1:]]].reset_index(drop=True)


def _load_development(input_dir: Path) -> tuple[DataFrame, dict[str, Any], str]:
    _, source_report, segments, segment_hashes = verify_source_artifact(input_dir)
    if not segments or len(segments[0]) <= A_HOLDOUT + INITIAL_TRAIN_ROWS:
        raise ValueError("stage_a_segment_too_short")
    candles = segments[0].iloc[:-A_HOLDOUT].copy().reset_index(drop=True)
    candles["date"] = pd.to_datetime(candles["date"], utc=True, errors="raise")
    if not (
        candles["date"].diff().dropna() == pd.Timedelta(seconds=HOUR)
    ).all():
        raise ValueError("stage_a_development_not_contiguous")
    values = candles[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values[:, 4] < 0):
        raise ValueError("stage_a_development_numeric_invalid")
    return candles, source_report, segment_hashes[0]


def _price_features(candles: DataFrame) -> DataFrame:
    close = pd.to_numeric(candles["close"], errors="raise")
    high = pd.to_numeric(candles["high"], errors="raise")
    low = pd.to_numeric(candles["low"], errors="raise")
    log_close = np.log(close)
    features = DataFrame(index=candles.index)
    features["log_return_1h"] = log_close.diff()
    for horizon in (6, 24, 48):
        features[f"momentum_{horizon}h"] = close.pct_change(
            horizon, fill_method=None
        )
    features["volatility_24h"] = (
        log_close.diff().rolling(24, min_periods=24).std() * math.sqrt(24)
    )
    for window in (50, 200):
        features[f"close_vs_sma_{window}"] = (
            close / close.rolling(window, min_periods=window).mean() - 1.0
        )
    change = close.diff()
    gain = change.clip(lower=0.0).rolling(14, min_periods=14).mean()
    loss = (-change.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    denominator = gain + loss
    features["rsi_14"] = (100.0 * gain / denominator.where(denominator > 0)).mask(
        denominator.eq(0), 50.0
    )
    rolling_high = high.rolling(24, min_periods=24).max()
    rolling_low = low.rolling(24, min_periods=24).min()
    width = rolling_high - rolling_low
    features["range_position_24h"] = (
        (close - rolling_low) / width.where(width > 0)
    ).mask(width.eq(0) & close.eq(rolling_low), 0.5)
    return features.replace([np.inf, -np.inf], np.nan)


def _event_outcomes(candles: DataFrame) -> tuple[dict[tuple[int, float, float], DataFrame], list[dict[str, Any]]]:
    close = pd.to_numeric(candles["close"], errors="raise")
    volatility = np.log(close).diff().rolling(24, min_periods=24).std() * math.sqrt(24)
    all_outcomes: dict[tuple[int, float, float], DataFrame] = {}
    scenarios: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        event_indices = [
            index
            for index in range(24, len(candles) - horizon, horizon)
            if math.isfinite(float(volatility.iloc[index]))
            and float(volatility.iloc[index]) > 0.0
        ]
        if not event_indices:
            raise ValueError(f"triple_barrier_event_grid_empty:{horizon}")
        events = DataFrame(
            {
                "event_index": event_indices,
                "side": [1] * len(event_indices),
                "target_volatility": [
                    float(volatility.iloc[index]) for index in event_indices
                ],
            }
        )
        for profit_multiplier in PT_MULTIPLIERS:
            for stop_multiplier in SL_MULTIPLIERS:
                config = TripleBarrierConfig(
                    profit_taking_multiplier=profit_multiplier,
                    stop_loss_multiplier=stop_multiplier,
                    max_holding_bars=horizon,
                    fee_bps_per_side=0.0,
                    slippage_bps_per_side=0.0,
                    timeframe_seconds=HOUR,
                )
                outcomes = apply_triple_barrier(candles, events, config)
                all_outcomes[(horizon, profit_multiplier, stop_multiplier)] = outcomes
                gross = outcomes["gross_return"].to_numpy(float)
                for slippage in SLIPPAGE_BPS:
                    cost = 2.0 * (FEE_BPS_SIDE + slippage) / 10_000.0
                    scenarios.append(
                        {
                            "horizon_hours": horizon,
                            "profit_taking_multiplier": profit_multiplier,
                            "stop_loss_multiplier": stop_multiplier,
                            "slippage_bps_per_side": slippage,
                            "round_trip_cost_bps": 2.0 * (FEE_BPS_SIDE + slippage),
                            "scheduled_event_count_not_trade_count": len(outcomes),
                            "take_profit_count": int(outcomes["exit_reason"].eq("take_profit").sum()),
                            "stop_loss_count": int(outcomes["exit_reason"].eq("stop_loss").sum()),
                            "vertical_count": int(outcomes["exit_reason"].eq("vertical").sum()),
                            "mean_gross_return": float(np.mean(gross)),
                            "median_gross_return": float(np.median(gross)),
                            "mean_net_return": float(np.mean(gross - cost)),
                            "status": "DESCRIPTIVE_BARRIER_OUTCOMES_ONLY",
                        }
                    )
    return all_outcomes, scenarios


def _liquidity_capacity(candles: DataFrame, volume_unit: str) -> tuple[dict[str, Any], np.ndarray]:
    volume = pd.to_numeric(candles["volume"], errors="raise").to_numpy(float)
    close = pd.to_numeric(candles["close"], errors="raise").to_numpy(float)
    positive = volume > 0.0
    if volume_unit == "base_asset":
        quote_notional = volume * close
    elif volume_unit == "quote_asset":
        quote_notional = volume.copy()
    else:
        quote_notional = np.zeros_like(volume)
    cap = quote_notional * PARTICIPATION_CAP
    if not positive.any():
        status = "BLOCKED_ZERO_SOURCE_VOLUME"
    elif volume_unit == "unknown":
        status = "BLOCKED_VOLUME_UNIT_UNVERIFIED"
    else:
        status = "READY_FOR_CAP_CHECK"
    return (
        {
            "source_volume_column": "volume",
            "source_volume_unit": volume_unit,
            "positive_volume_rows": int(positive.sum()),
            "zero_volume_rows": int((~positive).sum()),
            "liquidity_participation_cap_fraction": PARTICIPATION_CAP,
            "max_order_notional_quote_currency": float(cap.max()),
            "rows_with_positive_order_capacity": int((cap > 0.0).sum()),
            "status": status,
        },
        cap,
    )


def _fold_boundaries(row_count: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + OOS_WINDOW_ROWS, row_count))
        for start in range(INITIAL_TRAIN_ROWS, row_count, OOS_WINDOW_ROWS)
    ]


def _fit_predict(train_x: DataFrame, train_y: np.ndarray, test_x: DataFrame) -> np.ndarray:
    from lightgbm import LGBMClassifier  # type: ignore[import-not-found]

    if np.unique(train_y).size != 2:
        raise ValueError("triple_barrier_training_fold_single_class")
    model = LGBMClassifier(
        objective="binary",
        n_estimators=120,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=40,
        reg_lambda=2.0,
        random_state=17,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(train_x, train_y)
    positive_class = list(model.classes_).index(1)
    probabilities = np.asarray(model.predict_proba(test_x)[:, positive_class], dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("triple_barrier_oos_probabilities_non_finite")
    return probabilities


def _evaluate_candidate(
    candles: DataFrame,
    outcomes: DataFrame,
    features: DataFrame,
    capacity: np.ndarray,
    candidate_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_indices = outcomes["event_index"].to_numpy(dtype=int)
    event_features = features.iloc[event_indices].reset_index(drop=True)
    usable = np.isfinite(event_features.to_numpy(dtype=float)).all(axis=1)
    labelled = outcomes.loc[usable].reset_index(drop=True)
    event_features = event_features.loc[usable].reset_index(drop=True)
    folds: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fold_index, (test_start, test_stop) in enumerate(_fold_boundaries(len(candles))):
        train_mask = (
            labelled["event_index"].lt(test_start)
            & labelled["label_end_index"].lt(test_start)
        )
        test_mask = (
            labelled["event_index"].ge(test_start)
            & labelled["event_index"].lt(test_stop)
            & labelled["label_end_index"].lt(test_stop)
        )
        train_events = labelled.loc[train_mask].reset_index(drop=True)
        test_events = labelled.loc[test_mask].reset_index(drop=True)
        train_x = event_features.loc[train_mask].reset_index(drop=True)
        test_x = event_features.loc[test_mask].reset_index(drop=True)
        if len(train_events) < 50 or test_events.empty:
            folds.append(
                {
                    "candidate_id": candidate_id,
                    "fold_index": fold_index,
                    "status": "SKIPPED_INSUFFICIENT_EVENTS",
                    "train_events": len(train_events),
                    "test_events": len(test_events),
                }
            )
            continue
        train_labels = train_events["barrier_label"].eq(1).to_numpy(dtype=np.int8)
        if np.unique(train_labels).size != 2:
            folds.append(
                {
                    "candidate_id": candidate_id,
                    "fold_index": fold_index,
                    "status": "SKIPPED_SINGLE_CLASS_TRAIN",
                    "train_events": len(train_events),
                    "test_events": len(test_events),
                }
            )
            continue
        probabilities = _fit_predict(train_x, train_labels, test_x)
        for offset, (_, event) in enumerate(test_events.iterrows()):
            candle_index = int(event["event_index"])
            probability = float(probabilities[offset])
            signalled = probability >= PROBABILITY_THRESHOLD
            order_cap = float(capacity[candle_index])
            predictions.append(
                {
                    "candidate_id": candidate_id,
                    "fold_index": fold_index,
                    "event_index": candle_index,
                    "event_timestamp_utc": utc_text(candles["date"].iloc[candle_index]),
                    "label_end_index": int(event["label_end_index"]),
                    "predicted_probability_take_profit_first": probability,
                    "probability_threshold": PROBABILITY_THRESHOLD,
                    "realized_barrier_label": int(event["barrier_label"]),
                    "exit_reason": str(event["exit_reason"]),
                    "gross_return": float(event["gross_return"]),
                    "max_order_notional_quote_currency": order_cap,
                    "counterfactual_signal": bool(signalled),
                    "liquidity_eligible": bool(order_cap > 0.0),
                    "executable_trade": bool(signalled and order_cap > 0.0),
                }
            )
        folds.append(
            {
                "candidate_id": candidate_id,
                "fold_index": fold_index,
                "test_start_index": test_start,
                "test_stop_index": test_stop,
                "train_events": len(train_events),
                "test_events": len(test_events),
                "train_label_purge_pass": bool(
                    train_events["label_end_index"].lt(test_start).all()
                ),
                "status": "COMPLETED",
            }
        )
    return folds, predictions


def _drawdown(returns: Sequence[float]) -> float | None:
    if not returns:
        return None
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for value in returns:
        if not math.isfinite(value) or value <= -1.0:
            raise ValueError("triple_barrier_oos_return_invalid")
        equity *= 1.0 + value
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return abs(worst)


def _stress_metrics(predictions: list[dict[str, Any]], slippage: float) -> dict[str, Any]:
    round_trip_cost = 2.0 * (FEE_BPS_SIDE + slippage) / 10_000.0
    signalled = [row for row in predictions if row["counterfactual_signal"]]
    executable = [row for row in signalled if row["executable_trade"]]

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        returns = [float(row["gross_return"]) - round_trip_cost for row in rows]
        count = len(returns)
        rate = sum(value > 0.0 for value in returns) / count if count else None
        return {
            "trade_count": count,
            "win_rate": rate,
            "net_expectancy_per_trade": float(np.mean(returns)) if count else None,
            "max_drawdown_abs": _drawdown(returns),
            "compounded_return": float(np.prod(1.0 + np.asarray(returns)) - 1.0) if count else 0.0,
        }

    actual = summary(executable)
    gates = {
        "trade_count_pass": actual["trade_count"] >= 100,
        "win_rate_pass": actual["win_rate"] is not None and actual["win_rate"] > 0.60,
        "expectancy_pass": (
            actual["net_expectancy_per_trade"] is not None
            and actual["net_expectancy_per_trade"] > 0.0
        ),
        "drawdown_pass": (
            actual["max_drawdown_abs"] is not None and actual["max_drawdown_abs"] < 0.15
        ),
    }
    return {
        "slippage_bps_per_side": slippage,
        "round_trip_total_cost_bps": 2.0 * (FEE_BPS_SIDE + slippage),
        "counterfactual_signals_not_eligible_for_acceptance": summary(signalled),
        "executable_trade_metrics": actual,
        "strict_economic_gate_on_executable_trades": {**gates, "passed": all(gates.values())},
    }


def run_experiment(
    input_dir: Path,
    output_dir: Path,
    commit_sha: str,
    orderbook_csv: Path | None,
    volume_unit: str,
) -> dict[str, Any]:
    if len(commit_sha) != 40 or any(char not in "0123456789abcdef" for char in commit_sha):
        raise ValueError("research_commit_sha_invalid")
    candles, source_report, segment_hash = _load_development(input_dir)
    outcomes_by_config, barrier_scenarios = _event_outcomes(candles)
    liquidity, capacity = _liquidity_capacity(candles, volume_unit)
    blockers: list[str] = []
    orderbook_status = "NOT_PROVIDED"
    book_features: DataFrame | None = None
    book_rows: int | None = None
    if orderbook_csv is not None and orderbook_csv.is_file():
        book = pd.read_csv(orderbook_csv)
        book_rows = len(book)
        try:
            book_features = add_relative_orderbook_features(candles, book)
            orderbook_status = "VALIDATED_RELATIVE_FEATURES"
        except (ValueError, KeyError, TypeError, pd.errors.ParserError) as error:
            orderbook_status = "INVALID"
            blockers.append(f"orderbook_data_invalid:{error}")
    else:
        blockers.append("historical_binance_orderbook_snapshots_not_provided")
    if liquidity["status"] != "READY_FOR_CAP_CHECK":
        blockers.append(str(liquidity["status"]).lower())
    if int(liquidity["rows_with_positive_order_capacity"]) == 0:
        blockers.append("liquidity_filter_blocks_all_orders")

    price_features = _price_features(candles)
    feature_groups: dict[str, DataFrame] = {"ohlcv_only": price_features}
    if book_features is not None:
        feature_groups["ohlcv_plus_binance_relative_book"] = pd.concat(
            [price_features.reset_index(drop=True), book_features.drop(columns=["date"])],
            axis=1,
        )
    all_folds: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    for (horizon, profit_multiplier, stop_multiplier), outcomes in outcomes_by_config.items():
        candidate_id = (
            f"ohlcv_only|h{horizon}|pt{profit_multiplier:g}|sl{stop_multiplier:g}"
        )
        for group_name, matrix in feature_groups.items():
            current_id = candidate_id.replace("ohlcv_only", group_name, 1)
            folds, predictions = _evaluate_candidate(
                candles, outcomes, matrix, capacity, current_id
            )
            all_folds.extend(folds)
            all_predictions.extend(predictions)
            for slippage in SLIPPAGE_BPS:
                stress_rows.append(
                    {
                        "candidate_id": current_id,
                        "feature_group": group_name,
                        "horizon_hours": horizon,
                        "profit_taking_multiplier": profit_multiplier,
                        "stop_loss_multiplier": stop_multiplier,
                        **_stress_metrics(predictions, slippage),
                    }
                )
    economic_ready = (
        book_features is not None
        and liquidity["status"] == "READY_FOR_CAP_CHECK"
        and int(liquidity["rows_with_positive_order_capacity"]) > 0
    )
    passing = {
        row["candidate_id"]
        for row in stress_rows
        if row["feature_group"] == "ohlcv_plus_binance_relative_book"
        and row["strict_economic_gate_on_executable_trades"]["passed"]
    }
    economic_status = (
        "NOT_EVALUATED_BLOCKED_INPUT_DATA"
        if not economic_ready
        else "NO_GO_THRESHOLDS_NOT_MET"
        if not passing
        else "THRESHOLDS_MET_RESEARCH_ONLY"
    )
    report: dict[str, Any] = {
        "report_schema": SCHEMA,
        "run_status": "BLOCKED_INPUT_DATA" if blockers else "COMPLETED_RESEARCH_ONLY",
        "created_by_commit_sha": commit_sha,
        "source": {
            "coverage_workflow_run_id": SOURCE_RUN_ID,
            "coverage_csv_sha256": source_report["csv_sha256"],
            "coverage_manifest_sha256": source_report["manifest_sha256"],
            "segment_a_rows_sha256": segment_hash,
            "symbol": "BTCUSDT",
            "ohlcv_venue": "Nobitex",
            "timeframe": "1h",
            "development_rows": len(candles),
            "reserved_final_90_day_holdout_rows": A_HOLDOUT,
            "holdout_used": False,
            "development_start_utc": utc_text(candles["date"].iloc[0]),
            "development_end_utc": utc_text(candles["date"].iloc[-1]),
        },
        "protocol": {
            "horizons_hours": list(HORIZONS),
            "profit_taking_multipliers": list(PT_MULTIPLIERS),
            "stop_loss_multipliers": list(SL_MULTIPLIERS),
            "barrier_candidate_count": len(outcomes_by_config),
            "model_candidate_count_planned": 2 * len(outcomes_by_config),
            "model_candidate_count_executed": len(feature_groups) * len(outcomes_by_config),
            "model_candidate_count_blocked_for_missing_book": (
                0 if book_features is not None else len(outcomes_by_config)
            ),
            "diagnostic_scenario_count": len(barrier_scenarios),
            "oos_prediction_row_count": len(all_predictions),
            "fee_bps_per_side": FEE_BPS_SIDE,
            "slippage_bps_per_side": list(SLIPPAGE_BPS),
            "round_trip_total_cost_bps": [
                2.0 * (FEE_BPS_SIDE + slip) for slip in SLIPPAGE_BPS
            ],
            "liquidity_participation_cap_fraction": PARTICIPATION_CAP,
            "orderbook_venue": "Binance USD-M Futures BTCUSDT",
            "orderbook_feature_policy": (
                "Only normalized top-5/top-20 imbalance and relative best-level spread "
                "are used; absolute depth is not used for Nobitex capacity or as a model feature."
            ),
            "book_snapshot_max_age_seconds": BOOK_MAX_AGE_SECONDS,
            "book_join": "backward point-in-time join; stale or missing snapshot fails closed",
            "probability_threshold_fixed_not_tuned": PROBABILITY_THRESHOLD,
        },
        "data_availability": {
            "orderbook_status": orderbook_status,
            "orderbook_csv": str(orderbook_csv) if orderbook_csv else None,
            "orderbook_rows": book_rows,
            "liquidity": liquidity,
            "blockers": blockers,
        },
        "barrier_outcome_diagnostics_not_execution_backtest": barrier_scenarios,
        "walk_forward": {
            "feature_groups": list(feature_groups),
            "fold_metrics": all_folds,
            "candidate_cost_stress_metrics": stress_rows,
            "purge": "training label_end_index < test_start; test label_end_index < test_stop",
            "counterfactual_predictions_are_not_economic_evidence": True,
        },
        "economic_evaluation": {
            "status": economic_status,
            "executable_trade_count": sum(
                1 for row in all_predictions if row["executable_trade"]
            ),
            "candidate_pass_count": len(passing),
            "required_oos_trade_count": ">=100 executable trades",
            "required_win_rate": ">60%",
            "required_net_expectancy": ">0 after costs",
            "required_absolute_max_drawdown": "<15%",
            "all_gates_must_pass": True,
        },
        "cross_venue_limitations": [
            "Binance Order Book is used only for relative directional imbalance and relative spread.",
            "Absolute Binance depth is never interpreted as Nobitex depth or liquidity.",
            "Order size is capped at 1% of contemporaneous, unit-verified Nobitex quote notional.",
            "All 58,655 source Segment A candles have zero volume, so current sample grants zero order capacity.",
            "The pinned OHLCV artifact contains no historical Binance Level-2 snapshots; no synthetic Order Book is generated.",
            "Stress costs are 25 bps per-side fee plus 20/35/50 bps per-side slippage, for 90/120/150 bps round trip.",
        ],
        "safety_invariants": dict(SAFETY),
    }
    if report["safety_invariants"] != SAFETY:
        raise ValueError("research_safety_invariant_violation")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "triple-barrier-orderbook-report.json", report)
    _write_csv(output_dir / "barrier-outcome-diagnostics.csv", barrier_scenarios)
    _write_csv(output_dir / "oos-predictions.csv", all_predictions)
    _write_csv(output_dir / "oos-candidate-cost-stress.csv", stress_rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--orderbook-csv", type=Path, default=None)
    parser.add_argument(
        "--nobitex-volume-unit",
        choices=("unknown", "base_asset", "quote_asset"),
        default="unknown",
    )
    args = parser.parse_args()
    report = run_experiment(
        args.input_dir,
        args.output_dir,
        args.commit_sha,
        args.orderbook_csv,
        args.nobitex_volume_unit,
    )
    print(
        json.dumps(
            {
                "commit_sha": report["created_by_commit_sha"],
                "run_status": report["run_status"],
                "barrier_scenarios": report["protocol"]["diagnostic_scenario_count"],
                "executed_model_candidates": report["protocol"]["model_candidate_count_executed"],
                "oos_predictions": report["protocol"]["oos_prediction_row_count"],
                "orderbook_status": report["data_availability"]["orderbook_status"],
                "liquidity_status": report["data_availability"]["liquidity"]["status"],
                "economic_status": report["economic_evaluation"]["status"],
                "Champion": report["safety_invariants"]["Champion"],
                "order_permission": report["safety_invariants"]["order_permission"],
                "trading_authorization": report["safety_invariants"]["trading_authorization"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
