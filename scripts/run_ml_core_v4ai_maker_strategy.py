"""V4-AN ETHUSDT 30m HMM regime-adaptive economic validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import scripts.run_ml_core_v4ac_cost_optimization as v4ac
import scripts.run_ml_core_v4af_meta_final as v4af
from services.meta_labeling import v4ah as advanced
from services.meta_labeling import v4am as regime_engine
from services.meta_labeling.v4x import (
    V4X_OOF_FOLD_COUNT,
    generate_oof_primary_events,
)
from services.ml_core.economic_backtest import simulate_signals

SCHEMA = "alix.ml-core-v4an-regime-adaptive.v1"
SYMBOL = "ETHUSDT"
TIMEFRAME = "30m"
DAYS = 1825
ALLOW_EXCHANGE_GAPS = True
INITIAL_EQUITY = 10_000.0
FEE_BPS_PER_SIDE = 10.0
SLIPPAGE_BPS_PER_SIDE = 5.0
ROUND_TRIP_COST_BPS = 30.0
ADX_FEATURE = "%-trend_adx_14"
ADX_PERIOD = 14
ADX_THRESHOLD = 20.0
RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 45.0
EMA_FAST_PERIOD = 50
EMA_SLOW_PERIOD = 200
RSI_FEATURE = "%-primary_rsi_14"
EMA50_DISTANCE_FEATURE = "%-primary_close_vs_ema50_pct"
EMA_DISTANCE_FEATURE = "%-primary_close_vs_ema200_pct"
EMA_SPREAD_FEATURE = "%-primary_ema50_vs_ema200_pct"
PRIMARY_SIGNAL_GENERATION = "HMM3_REGIME_ADAPTIVE_EMA50_RSI45_NO_CUSUM"
RESEARCH_SIGNAL_COUNT_TARGET = 100
LEGACY_PRIMARY_FEATURE_COLUMNS = (
    "%-primary_bollinger_bandwidth_pct",
    "%-primary_close_vs_bollinger_upper_pct",
    "%-primary_close_vs_ema200_pct",
    "%-primary_ema200_slope_10_pct",
    "%-primary_breakout_distance_pct",
)
REGIME_WARMUP_BARS = 200
REGIME_CLUSTER_COUNT = 3
HMM_FEATURE_WINDOW_BARS = 48
HMM_REALIZED_VOLATILITY_FEATURE = "%-v4an_realized_volatility_48"
RANGE_RSI_ENTRY_THRESHOLD = 45.0
TECHNICAL_FEATURE_GENERATION = "CAUSAL_WILDER_RSI14_EMA50_EMA200_FEATURES_ONLY"
HMM_FEATURE_GENERATION = "CLOSED_BAR_LOG_RETURN_AND_TRAILING_48_BAR_REALIZED_VOLATILITY"
META_TARGET_ACCEPT_FRACTION = 0.70
MINIMUM_TRADES = 30
MINIMUM_NET_PF = 1.5
MINIMUM_WIN_RATE = 0.45
MAXIMUM_DRAWDOWN = 15.0
DSR_MINIMUM_RETURN_OBSERVATIONS = 4
SORTINO_TARGET_RETURN_PER_TRADE = 0.0
BASE_CONTROL = "PRIMARY_HMM_REGIME_ADAPTIVE_30M_30BPS"
ADX_ABLATION = "PRIMARY_ADX20_DIAGNOSTIC_ABLATION_HMM_30M_30BPS"
META_CANDIDATE = "META_TOP_70_HMM_REGIME_ADAPTIVE_30M_30BPS"
CANDIDATE_NAMES = (BASE_CONTROL, ADX_ABLATION, META_CANDIDATE)
PRIMARY_FILTER_CONTRACT: dict[str, str] = {
    "event_filter": "EVERY_CLOSED_30M_BAR_NO_CUSUM_EVENT_FILTER",
    "regime_model": "DIAGONAL_GAUSSIAN_HMM_3_STATE",
    "regime_state_selection": "DOMINANT_POSTERIOR_HARD_SELECTOR",
    "momentum_strategy": "CLOSE_ABOVE_EMA50_TRENDING_UP_ONLY_SPOT_LONG",
    "mean_reversion_strategy": "WILDER_RSI14_LT45_RANGING_ONLY",
    "downtrend_policy": "SPOT_LONG_ONLY_NO_ENTRY",
    "position_sizing": "INVERSE_24H_REALIZED_VOL_48_BAR_OUTER_TRAIN_MEDIAN_CAPPED_AT_1X",
}

LABEL_COLUMNS = {
    "event_index",
    "fold_id",
    "label_end_idx",
    "meta_label",
    "net_pnl_usd",
    "net_return",
    "exit_reason",
    "primary_signal",
    "primary_oof",
}
PRIMARY_GEOMETRY: dict[str, float] = {
    "initial_stop_atr": 1.5,
    "take_profit_atr": 2.5,
    "trailing_activation_atr": 1.5,
    "trailing_step_atr": 2.0,
    "horizon_bars": 48.0,
}
EXIT_POLICIES: tuple[dict[str, float | str], ...] = (
    {
        "name": "V4AI_REFERENCE",
        "initial_stop_atr": 1.5,
        "take_profit_atr": 2.5,
        "trailing_activation_atr": 1.0,
        "trailing_step_atr": 1.5,
        "horizon_bars": 48.0,
    },
    {
        "name": "V4AE_HISTORICAL_BEST",
        "initial_stop_atr": 1.5,
        "take_profit_atr": 2.5,
        "trailing_activation_atr": 1.5,
        "trailing_step_atr": 2.0,
        "horizon_bars": 48.0,
    },
    {
        "name": "V4AF_OWNER_GEOMETRY",
        "initial_stop_atr": 1.5,
        "take_profit_atr": 2.5,
        "trailing_activation_atr": 1.5,
        "trailing_step_atr": 1.5,
        "horizon_bars": 48.0,
    },
)


def _validate_request(days: int, commit_sha: str) -> None:
    valid_sha = len(commit_sha) == 40 and all(
        character in "0123456789abcdef" for character in commit_sha
    )
    if days != DAYS or not valid_sha:
        raise ValueError("v4ai_request_identity_invalid")
    if v4af.SYMBOL != SYMBOL or TIMEFRAME not in v4af.TIMEFRAME_MINUTES:
        raise ValueError("v4ai_market_scope_invalid")
    _validate_cost_contract(_execution_config())


def _validate_cost_contract(execution: Any) -> None:
    if not (
        math.isclose(float(execution.fee_bps), FEE_BPS_PER_SIDE)
        and math.isclose(float(execution.slippage_bps), SLIPPAGE_BPS_PER_SIDE)
        and math.isclose(
            2.0 * (float(execution.fee_bps) + float(execution.slippage_bps)),
            ROUND_TRIP_COST_BPS,
        )
    ):
        raise ValueError("v4ai_maker_cost_contract_invalid")


def _execution_config() -> Any:
    execution = replace(
        v4ac._execution_config(),
        fee_bps=FEE_BPS_PER_SIDE,
        slippage_bps=SLIPPAGE_BPS_PER_SIDE,
    )
    _validate_cost_contract(execution)
    return execution


def _trade_management_config(policy: dict[str, Any] | None = None) -> Any:
    selected = PRIMARY_GEOMETRY if policy is None else policy
    base = v4af._trade_management_config(None)
    return replace(
        base,
        stop_loss_atr_multiplier=float(selected["initial_stop_atr"]),
        take_profit_atr_multiplier=float(selected["take_profit_atr"]),
        trailing_stop_enabled=True,
        trailing_activation_atr=float(selected["trailing_activation_atr"]),
        trailing_step_atr=float(selected["trailing_step_atr"]),
        horizon_candles=int(selected["horizon_bars"]),
    )


def _validated_primary_close(bars: DataFrame) -> Series:
    if "close" not in bars.columns:
        raise ValueError("v4ai_primary_close_column_missing")
    close = pd.to_numeric(bars["close"], errors="coerce").reset_index(drop=True)
    values = close.to_numpy(dtype=float)
    if not np.isfinite(values).all() or bool((close <= 0.0).any()):
        raise ValueError("v4ai_primary_close_values_invalid")
    return close


def _causal_wilder_rsi(close: Series) -> Series:
    changes = close.diff()
    gains = changes.clip(lower=0.0)
    losses = -changes.clip(upper=0.0)
    average_gain = gains.ewm(
        alpha=1.0 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD,
    ).mean()
    average_loss = losses.ewm(
        alpha=1.0 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD,
    ).mean()
    average_movement = average_gain + average_loss
    rsi = 100.0 * average_gain / average_movement.where(average_movement > 0.0)
    return rsi.mask(average_movement.eq(0.0), 50.0)


def _finite_primary_feature_rows(features: DataFrame) -> np.ndarray:
    numeric = features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return cast(np.ndarray, np.isfinite(numeric).all(axis=1))


def _ema_trend_rsi_primary_features(
    bars: DataFrame,
    base_features: DataFrame,
) -> tuple[DataFrame, Series, dict[str, Any]]:
    if len(bars) != len(base_features) or bars.empty:
        raise ValueError("v4ai_primary_feature_alignment_invalid")
    if not base_features.columns.is_unique:
        raise ValueError("v4ai_primary_feature_columns_invalid")
    if not set(LEGACY_PRIMARY_FEATURE_COLUMNS).issubset(base_features.columns):
        raise ValueError("v4ai_legacy_breakout_feature_schema_invalid")

    close = _validated_primary_close(bars)
    rsi = _causal_wilder_rsi(close)
    ema_fast = close.ewm(
        span=EMA_FAST_PERIOD,
        adjust=False,
        min_periods=EMA_FAST_PERIOD,
    ).mean()
    ema_slow = close.ewm(
        span=EMA_SLOW_PERIOD,
        adjust=False,
        min_periods=EMA_SLOW_PERIOD,
    ).mean()

    features = (
        base_features.drop(columns=list(LEGACY_PRIMARY_FEATURE_COLUMNS))
        .reset_index(drop=True)
        .copy()
    )
    features[RSI_FEATURE] = rsi
    features[EMA50_DISTANCE_FEATURE] = close / ema_fast - 1.0
    features[EMA_DISTANCE_FEATURE] = close / ema_slow - 1.0
    features[EMA_SPREAD_FEATURE] = ema_fast / ema_slow - 1.0

    valid_rsi = np.isfinite(rsi.to_numpy(dtype=float))
    valid_ema_fast = np.isfinite(ema_fast.to_numpy(dtype=float))
    valid_ema_slow = np.isfinite(ema_slow.to_numpy(dtype=float))
    valid_ema = valid_ema_fast & valid_ema_slow
    valid_features = _finite_primary_feature_rows(features)
    entry_oversold = rsi.lt(RSI_ENTRY_THRESHOLD)
    trend_aligned = close.gt(ema_fast)
    signal = (
        (
            entry_oversold
            & trend_aligned
            & Series(valid_rsi)
            & Series(valid_ema)
            & Series(valid_features)
        )
        .fillna(False)
        .astype(bool)
    )
    diagnostics = {
        "generation": TECHNICAL_FEATURE_GENERATION,
        "rsi_method": "WILDER_EMA_ALPHA_1_OVER_14_ADJUST_FALSE",
        "rsi_period": RSI_PERIOD,
        "rsi_threshold": RSI_ENTRY_THRESHOLD,
        "rsi_comparison": "STRICT_LT",
        "ema_fast_period": EMA_FAST_PERIOD,
        "ema_slow_period": EMA_SLOW_PERIOD,
        "ema_method": "CAUSAL_SPAN_ADJUST_FALSE",
        "trend_comparison": "CLOSE_STRICTLY_ABOVE_EMA50",
        "rsi_valid_rows": int(valid_rsi.sum()),
        "ema_fast_valid_rows": int(valid_ema_fast.sum()),
        "ema_slow_valid_rows": int(valid_ema_slow.sum()),
        "ema_valid_rows": int(valid_ema.sum()),
        "feature_valid_rows": int(valid_features.sum()),
        "oversold_rows": int((entry_oversold & Series(valid_rsi)).sum()),
        "trend_aligned_rows": int((trend_aligned & Series(valid_ema_fast)).sum()),
        "primary_buy_count": int(signal.sum()),
        "research_signal_count_target": RESEARCH_SIGNAL_COUNT_TARGET,
        "research_signal_target_met": int(signal.sum()) >= RESEARCH_SIGNAL_COUNT_TARGET,
        "lookahead_used": False,
        "execution_timing": "CLOSED_30M_BAR_NEXT_BAR_OPEN",
    }
    signal.name = "primary_buy_signal"
    return features, signal, diagnostics


def _generate_primary_oof_events(
    primary_signal: Series,
    start_index: int,
    stop_index: int,
) -> DataFrame:
    events = generate_oof_primary_events(
        primary_signal,
        start_index=start_index,
        stop_index=stop_index,
        fold_count=V4X_OOF_FOLD_COUNT,
    )
    events["primary_signal_generation"] = PRIMARY_SIGNAL_GENERATION
    return events


def _apply_adx_ablation(
    base_signal: Series,
    features: DataFrame,
) -> tuple[Series, dict[str, Any]]:
    if len(base_signal) != len(features) or ADX_FEATURE not in features:
        raise ValueError("v4ai_adx_feature_alignment_invalid")
    adx = pd.to_numeric(features[ADX_FEATURE], errors="coerce").reset_index(drop=True)
    signal = base_signal.reset_index(drop=True)
    if not pd.api.types.is_bool_dtype(signal.dtype) or signal.isna().any():
        raise ValueError("v4ai_primary_signal_invalid")
    valid_adx = np.isfinite(adx.to_numpy(dtype=float))
    filtered = signal & adx.gt(ADX_THRESHOLD) & Series(valid_adx)
    diagnostics = {
        "filter": "ADX_14_STRICTLY_GREATER_THAN_20",
        "feature": ADX_FEATURE,
        "period": ADX_PERIOD,
        "threshold": ADX_THRESHOLD,
        "comparison": "STRICT_GT",
        "base_primary_buy_count": int(signal.sum()),
        "valid_adx_rows": int(valid_adx.sum()),
        "adx_qualified_rows": int((adx.gt(ADX_THRESHOLD) & Series(valid_adx)).sum()),
        "filtered_primary_buy_count": int(filtered.sum()),
    }
    return filtered.astype(bool), diagnostics


def _feature_columns(training: DataFrame) -> list[str]:
    columns = [str(name) for name in training.columns if name not in LABEL_COLUMNS]
    if not columns or not training.loc[:, columns].columns.is_unique:
        raise ValueError("v4ai_feature_columns_invalid")
    return columns


def _label_one_oof_event(
    event: dict[str, Any],
    train_bars: DataFrame,
    index_by_timestamp: dict[pd.Timestamp, int],
    train_end: int,
) -> dict[str, Any] | None:
    event_index = int(event["event_index"])
    if event.get("primary_signal") != "BUY" or event.get("primary_oof") is not True:
        raise ValueError("v4ai_oof_labels_require_oof_primary_buy_events")
    if event_index + int(PRIMARY_GEOMETRY["horizon_bars"]) >= train_end:
        return None
    actions = v4ac._build_forward_actions(train_bars, [event_index], train_end)
    replay = simulate_signals(
        train_bars,
        actions,
        _execution_config(),
        trade_management_config=_trade_management_config(),
    )
    trades = cast(list[dict[str, Any]], replay["trades"])
    if len(trades) != 1:
        raise ValueError("v4ai_oof_event_trade_count_invalid")
    trade = trades[0]
    exit_timestamp = pd.Timestamp(str(trade["exit_timestamp"]))
    if exit_timestamp.tzinfo is None:
        raise ValueError("v4ai_oof_exit_timezone_invalid")
    exit_index = index_by_timestamp.get(exit_timestamp.tz_convert("UTC"))
    if exit_index is None or not event_index < exit_index < train_end:
        raise ValueError("v4ai_oof_label_not_matured_inside_train")
    net_pnl = float(trade["net_pnl_usd"])
    net_return = float(trade["net_return_fraction"])
    if not math.isfinite(net_pnl) or not math.isfinite(net_return):
        raise ValueError("v4ai_oof_label_outcome_non_finite")
    return {
        "event_index": event_index,
        "fold_id": int(event["fold_id"]),
        "label_end_idx": exit_index,
        "meta_label": int(net_pnl > 0.0),
        "net_pnl_usd": net_pnl,
        "net_return": net_return,
        "exit_reason": str(trade["exit_reason"]),
        "primary_signal": "BUY",
        "primary_oof": True,
    }


def _training_oof_labels(
    bars: DataFrame,
    oof_events: DataFrame,
    train_end: int,
) -> tuple[DataFrame, dict[str, Any]]:
    train_bars = bars.iloc[:train_end].reset_index(drop=True)
    dates = pd.to_datetime(train_bars["date"], utc=True, errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("v4ai_training_timestamps_invalid")
    index_by_timestamp = {
        pd.Timestamp(value): index for index, value in enumerate(dates.tolist())
    }
    records: list[dict[str, Any]] = []
    immature = 0
    for event in oof_events.to_dict(orient="records"):
        record = _label_one_oof_event(
            cast(dict[str, Any], event),
            train_bars,
            index_by_timestamp,
            train_end,
        )
        if record is None:
            immature += 1
        else:
            records.append(record)
    if not records:
        raise ValueError("v4ai_meta_oof_labels_empty")
    labelled = (
        DataFrame.from_records(records)
        .sort_values("event_index")
        .reset_index(drop=True)
    )
    if labelled["event_index"].duplicated().any():
        raise ValueError("v4ai_meta_oof_labels_duplicate")
    if (labelled["label_end_idx"] >= train_end).any():
        raise ValueError("v4ai_meta_label_crosses_outer_train")
    class_counts = labelled["meta_label"].value_counts().to_dict()
    diagnostics = {
        "oof_primary_event_count": int(len(oof_events)),
        "matured_costed_oof_label_count": int(len(labelled)),
        "immature_oof_event_count": immature,
        "oof_fold_count": V4X_OOF_FOLD_COUNT,
        "oof_fold_ids": sorted(int(value) for value in labelled["fold_id"].unique()),
        "label_class_counts": {
            "meta_failure": int(class_counts.get(0, 0)),
            "meta_success": int(class_counts.get(1, 0)),
        },
        "label_target": "NET_PNL_POSITIVE_AFTER_MAKER_FEE_AND_SLIPPAGE_PROXY",
        "label_execution_costs": {
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        },
        "label_geometry": dict(PRIMARY_GEOMETRY),
        "training_labels_matured_before_validation": bool(
            int(labelled["label_end_idx"].max()) < train_end
        ),
    }
    return labelled, diagnostics


def _oof_training_frame(
    bars: DataFrame,
    features: DataFrame,
    primary_signal: Series,
    train_end: int,
) -> tuple[DataFrame, dict[str, Any]]:
    oof_events = _generate_primary_oof_events(
        primary_signal,
        start_index=EMA_SLOW_PERIOD,
        stop_index=train_end,
    )
    labelled, diagnostics = _training_oof_labels(bars, oof_events, train_end)
    return v4af._join_event_features(features, labelled), diagnostics


def _support_shortfall(error: ValueError) -> bool:
    code = str(error)
    prefixes = (
        "v4ai_insufficient_primary_signals",
        "v4ah_fit_rows_insufficient",
        "v4ah_fit_rows_insufficient_after_purge",
        "v4ah_temperature_rows_insufficient_after_purge",
        "v4ah_binary_class_support_insufficient",
        "v4ah_shap_feature_support_insufficient",
        "v4ah_temperature_calibration_inputs_invalid",
        "v4ah_separation_inputs_invalid",
        "v4x_insufficient_meta_",
        "v4ai_meta_oof_labels_empty",
        "v4x_oof_primary_signals_empty",
        "v4ah_regime_training_support_insufficient",
    )
    return any(code.startswith(prefix) for prefix in prefixes)


def _select_top_percentile_threshold(
    probabilities: Any,
) -> tuple[float, dict[str, Any]]:
    values = np.asarray(probabilities, dtype=float)
    if (
        values.ndim != 1
        or values.size < 1
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("v4ai_threshold_probability_input_invalid")
    cutoff_percentile = 1.0 - META_TARGET_ACCEPT_FRACTION
    threshold = float(np.quantile(values, cutoff_percentile, method="linear"))
    selected = values >= threshold
    return threshold, {
        "status": "SELECTED_TOP_70_PERCENT_ACCEPTANCE",
        "method": "PURGED_OOF_THRESHOLD_SLICE_ORDER_STATISTIC",
        "cutoff_percentile": cutoff_percentile,
        "target_accept_fraction": META_TARGET_ACCEPT_FRACTION,
        "threshold": threshold,
        "threshold_rows": int(len(values)),
        "selected_rows": int(selected.sum()),
        "selected_fraction": float(selected.mean()),
        "expected_top_fraction_before_ties": META_TARGET_ACCEPT_FRACTION,
        "labels_used": False,
        "validation_probabilities_used": False,
        "selection_scope": "THIRD_PURGED_OOF_TRAINING_SLICE_ONLY",
        "tie_rule": "GREATER_THAN_OR_EQUAL_WITH_TIES_INCLUDED",
    }


def _fit_meta_model(
    bars: DataFrame,
    features: DataFrame,
    primary_signal: Series,
    train_end: int,
    train_gate: dict[str, Any],
) -> dict[str, Any]:
    if not bool(train_gate.get("passed")):
        return {
            "skip_reason": "v4ai_insufficient_primary_signals",
            "separation_diagnostics": {"status": "NOT_AVAILABLE_MODEL_NOT_FIT"},
        }
    try:
        training, training_diagnostics = _oof_training_frame(
            bars, features, primary_signal, train_end
        )
        fit_rows, temperature_rows, threshold_rows, split_diagnostics = (
            advanced.split_fit_temperature_threshold_with_fallback(
                training, outer_train_end=train_end
            )
        )
        feature_names = _feature_columns(training)
        model, selected, shap_rankings, fit_diagnostics = (
            advanced.fit_shap_selected_lightgbm(
                fit_rows.loc[:, feature_names],
                fit_rows["meta_label"],
                top_k=30,
            )
        )
        raw_temperature = advanced._positive_probability(
            model, temperature_rows.loc[:, selected]
        )
        identity_temperature = (
            split_diagnostics.get("temperature_calibration_mode")
            == advanced.V4AH_TEMPERATURE_MODE_IDENTITY
        )
        if identity_temperature:
            separation = {
                "status": "NOT_AVAILABLE_IDENTITY_TEMPERATURE_FALLBACK",
                "metric_scope": ("PURGED_OOF_TEMPERATURE_SLICE_OUT_OF_FIT_PREDICTIONS"),
                "probability_scope": "RAW_PRE_TEMPERATURE_CALIBRATION",
                "sample_count": int(len(temperature_rows)),
                "labels_used": False,
            }
            temperature = 1.0
            temperature_diagnostics = {
                "method": "IDENTITY_FALLBACK_RAW_PROBABILITIES",
                "temperature": 1.0,
                "optimizer_success": None,
                "optimizer_iterations": None,
                "calibration_rows": int(len(temperature_rows)),
                "calibration_labels_used": False,
                "initial_nll": None,
                "calibrated_nll": None,
                "fit_scope": "IDENTITY_NO_FIT_PURGED_SLICE_THIN_OR_DEGENERATE",
                "warning": advanced.V4AH_TEMPERATURE_IDENTITY_WARNING,
            }
        else:
            separation = advanced.separation_diagnostics(
                raw_temperature,
                temperature_rows["meta_label"].to_numpy(dtype=int),
            )
            temperature, temperature_diagnostics = advanced.fit_temperature_scaler(
                raw_temperature,
                temperature_rows["meta_label"].to_numpy(dtype=int),
            )
        raw_threshold = advanced._positive_probability(
            model, threshold_rows.loc[:, selected]
        )
        calibrated_threshold = advanced.apply_temperature_scaling(
            raw_threshold, temperature
        )
        threshold, threshold_diagnostics = _select_top_percentile_threshold(
            calibrated_threshold
        )
        return {
            "skip_reason": None,
            "model": model,
            "selected_features": selected,
            "shap_rankings": shap_rankings,
            "temperature": temperature,
            "threshold": threshold,
            "separation_diagnostics": separation,
            "fit_diagnostics": fit_diagnostics,
            "split_diagnostics": split_diagnostics,
            "training_diagnostics": training_diagnostics,
            "temperature_diagnostics": temperature_diagnostics,
            "threshold_diagnostics": threshold_diagnostics,
        }
    except ValueError as error:
        if not _support_shortfall(error):
            raise
        return {
            "skip_reason": str(error),
            "training_diagnostics": locals().get("training_diagnostics", {}),
            "split_diagnostics": cast(
                dict[str, Any], getattr(error, "diagnostics", {})
            ),
            "separation_diagnostics": {"status": "NOT_AVAILABLE_MODEL_NOT_FIT"},
        }


def _hmm_feature_inputs(
    bars: DataFrame,
    features: DataFrame,
    *,
    apply_end: int,
) -> tuple[DataFrame, dict[str, Any]]:
    if (
        len(bars) != len(features)
        or bars.empty
        or isinstance(apply_end, bool)
        or not isinstance(apply_end, int)
        or not 2 <= apply_end <= len(bars)
        or "close" not in bars.columns
    ):
        raise ValueError("v4an_hmm_feature_input_invalid")
    generated = {
        regime_engine.LOG_RETURN_FEATURE,
        regime_engine.REALIZED_VOLATILITY_FEATURE,
        HMM_REALIZED_VOLATILITY_FEATURE,
    }
    if generated.intersection(features.columns):
        raise ValueError("v4an_hmm_feature_collision")
    close = pd.to_numeric(bars["close"].iloc[:apply_end], errors="coerce").reset_index(
        drop=True
    )
    close_values = close.to_numpy(dtype=float)
    if not np.isfinite(close_values).all() or bool((close <= 0.0).any()):
        raise ValueError("v4an_hmm_close_values_invalid")
    log_returns = np.log(close).diff()
    realized_volatility = log_returns.rolling(
        HMM_FEATURE_WINDOW_BARS,
        min_periods=HMM_FEATURE_WINDOW_BARS,
    ).std(ddof=0)
    log_return_values = np.full(len(bars), np.nan, dtype=float)
    volatility_values = np.full(len(bars), np.nan, dtype=float)
    log_return_values[:apply_end] = log_returns.to_numpy(dtype=float)
    volatility_values[:apply_end] = realized_volatility.to_numpy(dtype=float)
    transformed = features.reset_index(drop=True).copy()
    transformed[regime_engine.LOG_RETURN_FEATURE] = log_return_values
    transformed[regime_engine.REALIZED_VOLATILITY_FEATURE] = volatility_values
    diagnostics = {
        "generation": HMM_FEATURE_GENERATION,
        "realized_volatility_window_bars": HMM_FEATURE_WINDOW_BARS,
        "valid_log_return_rows": int(np.isfinite(log_return_values[:apply_end]).sum()),
        "valid_realized_volatility_rows": int(
            np.isfinite(volatility_values[:apply_end]).sum()
        ),
        "apply_end_index_exclusive": apply_end,
        "final_holdout_features_unavailable": bool(
            np.isnan(log_return_values[apply_end:]).all()
            and np.isnan(volatility_values[apply_end:]).all()
        ),
        "lookahead_used": False,
        "cusum_used": False,
    }
    return transformed, diagnostics


def _regime_features(
    features: DataFrame,
    *,
    train_end: int,
    apply_end: int,
) -> tuple[DataFrame, dict[str, Any]]:
    transformed, diagnostics = regime_engine.build_hmm_regime_features(
        features, train_end=train_end, apply_end=apply_end
    )
    if HMM_REALIZED_VOLATILITY_FEATURE in transformed.columns:
        raise ValueError("v4an_hmm_feature_collision")
    transformed = transformed.rename(
        columns={
            regime_engine.REALIZED_VOLATILITY_FEATURE: HMM_REALIZED_VOLATILITY_FEATURE
        }
    )
    return transformed, diagnostics


def _regime_reactive_primary_signal(
    bars: DataFrame,
    features: DataFrame,
    *,
    apply_end: int,
) -> tuple[DataFrame, Series, dict[str, Any]]:
    required = {
        regime_engine.HMM_TRENDING_UP_FEATURE,
        regime_engine.HMM_TRENDING_DOWN_FEATURE,
        regime_engine.HMM_RANGING_FEATURE,
        RSI_FEATURE,
        EMA50_DISTANCE_FEATURE,
    }
    if (
        len(bars) != len(features)
        or not 2 <= apply_end <= len(bars)
        or not required.issubset(features.columns)
        or "close" not in bars.columns
    ):
        raise ValueError("v4an_selector_input_schema_invalid")
    close = pd.to_numeric(bars["close"], errors="coerce").reset_index(drop=True)
    close_values = close.to_numpy(dtype=float)
    if not np.isfinite(close_values[:apply_end]).all() or bool(
        (close.iloc[:apply_end] <= 0.0).any()
    ):
        raise ValueError("v4an_selector_market_values_invalid")
    hmm_columns = [
        regime_engine.HMM_TRENDING_UP_FEATURE,
        regime_engine.HMM_TRENDING_DOWN_FEATURE,
        regime_engine.HMM_RANGING_FEATURE,
    ]
    probabilities = (
        features.loc[:, hmm_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    valid_hmm = np.isfinite(probabilities).all(axis=1)
    dominant = np.full(len(features), -1, dtype=int)
    dominant[valid_hmm] = np.argmax(probabilities[valid_hmm], axis=1)
    rsi = pd.to_numeric(features[RSI_FEATURE], errors="coerce").to_numpy(dtype=float)
    ema50_distance = pd.to_numeric(
        features[EMA50_DISTANCE_FEATURE], errors="coerce"
    ).to_numpy(dtype=float)
    valid = valid_hmm & np.isfinite(rsi) & np.isfinite(ema50_distance)
    trending_up = valid & (dominant == 0)
    trending_down = valid & (dominant == 1)
    ranging = valid & (dominant == 2)
    momentum_signal = trending_up & (ema50_distance > 0.0)
    mean_reversion_signal = ranging & (rsi < RANGE_RSI_ENTRY_THRESHOLD)
    signal_values = np.zeros(len(features), dtype=bool)
    signal_values[:apply_end] = (
        momentum_signal[:apply_end] | mean_reversion_signal[:apply_end]
    )
    signal = Series(signal_values, dtype=bool, name="primary_buy_signal")
    diagnostics = {
        "generation": PRIMARY_SIGNAL_GENERATION,
        "selector_type": "HMM_REGIME_HARD_SELECTOR",
        "dominant_state_policy": "ARGMAX_POSTERIOR_EXACTLY_ONE_REGIME",
        "momentum_strategy": "CLOSE_ABOVE_EMA50_TRENDING_UP_ONLY_SPOT_LONG",
        "mean_reversion_strategy": "WILDER_RSI14_LT45_RANGING_ONLY",
        "downtrend_policy": "RISK_OFF_NO_LONG_ENTRY_SPOT_ONLY",
        "trend_ema_period": EMA_FAST_PERIOD,
        "range_rsi_period": RSI_PERIOD,
        "range_rsi_threshold": RANGE_RSI_ENTRY_THRESHOLD,
        "dominant_trending_up_rows": int(trending_up[:apply_end].sum()),
        "dominant_trending_down_risk_off_rows": int(trending_down[:apply_end].sum()),
        "dominant_ranging_rows": int(ranging[:apply_end].sum()),
        "momentum_signal_count": int(momentum_signal[:apply_end].sum()),
        "mean_reversion_signal_count": int(mean_reversion_signal[:apply_end].sum()),
        "primary_buy_count": int(signal.sum()),
        "hard_selector_enabled": True,
        "soft_blend_enabled": False,
        "short_entries_enabled": False,
        "no_cusum_filter": True,
        "lookahead_used": False,
        "execution_timing": "CLOSED_30M_BAR_NEXT_BAR_OPEN",
    }
    return features.reset_index(drop=True), signal, diagnostics


def _validate_volatility_target_inputs(
    features: DataFrame,
    train_end: int,
    event_indices: list[int],
) -> tuple[np.ndarray, float]:
    column = HMM_REALIZED_VOLATILITY_FEATURE
    if (
        column not in features.columns
        or isinstance(train_end, bool)
        or not isinstance(train_end, int)
        or not 1 <= train_end <= len(features)
    ):
        raise ValueError("v4am_volatility_target_input_invalid")
    if event_indices != sorted(set(event_indices)):
        raise ValueError("v4am_position_fraction_event_order_invalid")
    volatility = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=float)
    training_values = volatility[:train_end]
    training_values = training_values[
        np.isfinite(training_values) & (training_values > 0.0)
    ]
    if len(training_values) < regime_engine.HMM_MINIMUM_TRAIN_ROWS:
        raise ValueError("v4am_volatility_target_training_support_insufficient")
    target_volatility = float(np.median(training_values))
    if not math.isfinite(target_volatility) or target_volatility <= 0.0:
        raise ValueError("v4am_volatility_target_invalid")
    return volatility, target_volatility


def _event_inverse_volatility_fraction(
    index: int,
    volatility: np.ndarray,
    *,
    train_end: int,
    target_volatility: float,
) -> float:
    if index < train_end or index >= len(volatility):
        raise ValueError("v4am_position_fraction_event_window_invalid")
    current_volatility = float(volatility[index])
    if not math.isfinite(current_volatility) or current_volatility <= 0.0:
        raise ValueError("v4am_position_fraction_volatility_invalid")
    fraction = min(1.0, target_volatility / current_volatility)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("v4am_position_fraction_out_of_bounds")
    return fraction


def _volatility_target_position_fractions(
    features: DataFrame,
    *,
    train_end: int,
    event_indices: list[int],
) -> tuple[dict[int, float], dict[str, Any]]:
    volatility, target_volatility = _validate_volatility_target_inputs(
        features, train_end, event_indices
    )
    fractions = {
        index: _event_inverse_volatility_fraction(
            index,
            volatility,
            train_end=train_end,
            target_volatility=target_volatility,
        )
        for index in event_indices
    }
    values = list(fractions.values())
    training_rows = int(
        np.sum(np.isfinite(volatility[:train_end]) & (volatility[:train_end] > 0.0))
    )
    diagnostics = {
        "method": "INVERSE_24H_REALIZED_VOLATILITY",
        "target_reference": "MEDIAN_OF_OUTER_TRAIN_48_BAR_REALIZED_VOLATILITY",
        "target_volatility": target_volatility,
        "training_rows": training_rows,
        "sizing_cap": 1.0,
        "leverage_enabled": False,
        "event_count": len(fractions),
        "fraction_min": min(values) if values else None,
        "fraction_median": float(np.median(values)) if values else None,
        "fraction_max": max(values) if values else None,
    }
    return fractions, diagnostics


def _apply_position_fractions(
    actions: list[dict[str, Any]],
    position_fractions: dict[int, float],
) -> None:
    for action in actions:
        if action.get("signal") != "BUY":
            continue
        index = int(action["row_index"])
        fraction = position_fractions.get(index)
        if (
            fraction is None
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) <= 1.0
        ):
            raise ValueError("v4am_position_fraction_action_missing_or_invalid")
        action["position_fraction"] = float(fraction)


def _forward_probability_map(
    features: DataFrame,
    event_indices: list[int],
    selected_features: list[str],
    model: Any,
    temperature: float,
) -> dict[int, float]:
    if not event_indices:
        return {}
    rows = features.iloc[event_indices].loc[:, selected_features].reset_index(drop=True)
    raw = advanced._positive_probability(model, rows)
    calibrated = advanced.apply_temperature_scaling(raw, temperature)
    return {
        index: float(value)
        for index, value in zip(event_indices, calibrated, strict=True)
    }


def _candidate_id(mode: str) -> str:
    if mode not in CANDIDATE_NAMES:
        raise ValueError("v4ai_candidate_mode_invalid")
    return f"{SYMBOL}|{TIMEFRAME}|HMM_REGIME_ADAPTIVE_NO_CUSUM|{mode}"


def _profit_factor(metrics: dict[str, Any]) -> float:
    if bool(metrics.get("net_profit_factor_infinite_no_losses")):
        return math.inf
    value = metrics.get("net_profit_factor")
    return -math.inf if value is None else float(value)


def _expectancy(trades: list[dict[str, Any]]) -> float:
    returns = [float(item["net_return_fraction"]) for item in trades]
    return float(np.mean(returns) * INITIAL_EQUITY) if returns else 0.0


def _initial_trade_risk_metrics(count: int) -> dict[str, Any]:
    return {
        "return_observation_count": count,
        "sharpe_ratio_per_trade": None,
        "sharpe_status": "INSUFFICIENT_TRADE_RETURNS",
        "return_skewness": None,
        "return_kurtosis_non_excess": None,
        "sortino_ratio_per_trade": None,
        "sortino_target_return_per_trade": SORTINO_TARGET_RETURN_PER_TRADE,
        "sortino_status": "INSUFFICIENT_TRADE_RETURNS",
        "dsr_z_score": None,
        "dsr_probability": None,
        "dsr_expected_max_sharpe_per_trade": None,
        "dsr_trial_count": 0,
        "dsr_trial_scope": "NOT_COMPUTED",
        "dsr_status": "AWAITING_TRIAL_COHORT",
    }


def _sortino_trade_metrics(excess_returns: np.ndarray) -> tuple[float | None, str]:
    downside = np.minimum(excess_returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    if downside_deviation <= 0.0:
        return None, "ZERO_DOWNSIDE_DEVIATION"
    return float(np.mean(excess_returns) / downside_deviation), "COMPUTED"


def _return_shape_metrics(returns: np.ndarray, count: int) -> dict[str, Any]:
    empty_metrics = {
        "return_skewness": None,
        "return_kurtosis_non_excess": None,
    }
    if count < 4:
        return empty_metrics
    centered_returns = returns - float(np.mean(returns))
    second_moment = float(np.mean(np.square(centered_returns)))
    if second_moment <= 0.0 or not math.isfinite(second_moment):
        return empty_metrics
    third_moment = float(np.mean(np.power(centered_returns, 3)))
    fourth_moment = float(np.mean(np.power(centered_returns, 4)))
    biased_skewness = third_moment / second_moment**1.5
    adjusted_skewness = math.sqrt(count * (count - 1)) / (count - 2) * biased_skewness
    biased_excess_kurtosis = fourth_moment / second_moment**2 - 3.0
    adjusted_excess_kurtosis = (
        (count - 1)
        / ((count - 2) * (count - 3))
        * ((count + 1) * biased_excess_kurtosis + 6.0)
    )
    if not math.isfinite(adjusted_skewness) or not math.isfinite(
        adjusted_excess_kurtosis
    ):
        return empty_metrics
    return {
        "return_skewness": float(adjusted_skewness),
        "return_kurtosis_non_excess": float(adjusted_excess_kurtosis + 3.0),
    }


def _trade_risk_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray(
        [float(trade["net_return_fraction"]) for trade in trades], dtype=float
    )
    if returns.ndim != 1 or not np.isfinite(returns).all():
        raise ValueError("v4ai_risk_metric_returns_invalid")
    count = int(returns.size)
    metrics = _initial_trade_risk_metrics(count)
    if count < 2:
        return metrics
    excess_returns = returns - SORTINO_TARGET_RETURN_PER_TRADE
    sortino, sortino_status = _sortino_trade_metrics(excess_returns)
    metrics["sortino_ratio_per_trade"] = sortino
    metrics["sortino_status"] = sortino_status
    standard_deviation = float(np.std(returns, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        metrics["sharpe_status"] = "ZERO_RETURN_VOLATILITY"
        return metrics
    metrics.update(
        {
            "sharpe_ratio_per_trade": float(
                np.mean(excess_returns) / standard_deviation
            ),
            "sharpe_status": "COMPUTED",
        }
    )
    metrics.update(_return_shape_metrics(returns, count))
    return metrics


def _dsr_input_status(metrics: dict[str, Any], trial_sharpe_count: int) -> str | None:
    if trial_sharpe_count < 2:
        return "INSUFFICIENT_TRIAL_SHARPES"
    if (
        int(metrics.get("return_observation_count", 0))
        < DSR_MINIMUM_RETURN_OBSERVATIONS
    ):
        return "INSUFFICIENT_RETURN_OBSERVATIONS"
    required_values = (
        metrics.get("sharpe_ratio_per_trade"),
        metrics.get("return_skewness"),
        metrics.get("return_kurtosis_non_excess"),
    )
    if any(value is None for value in required_values):
        return "RETURN_DISTRIBUTION_UNAVAILABLE"
    return None


def _expected_maximum_sharpe(trial_values: np.ndarray) -> float:
    trial_count = int(trial_values.size)
    euler_mascheroni = 0.5772156649015329
    normal = NormalDist()
    expected_max_multiplier = (1.0 - euler_mascheroni) * normal.inv_cdf(
        1.0 - 1.0 / trial_count
    ) + euler_mascheroni * normal.inv_cdf(1.0 - 1.0 / (trial_count * math.e))
    return (
        float(np.mean(trial_values))
        + float(np.std(trial_values, ddof=1)) * expected_max_multiplier
    )


def _dsr_variance_term(
    return_skewness: float, return_kurtosis: float, observed_sharpe: float
) -> float:
    return (
        1.0
        - return_skewness * observed_sharpe
        + ((return_kurtosis - 1.0) / 4.0) * observed_sharpe**2
    )


def _deflated_sharpe_diagnostics(
    metrics: dict[str, Any], trial_sharpes: list[float], trial_scope: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dsr_z_score": None,
        "dsr_probability": None,
        "dsr_expected_max_sharpe_per_trade": None,
        "dsr_trial_count": len(trial_sharpes),
        "dsr_trial_scope": trial_scope,
        "dsr_status": "INSUFFICIENT_TRIAL_SHARPES",
    }
    input_status = _dsr_input_status(metrics, len(trial_sharpes))
    if input_status is not None:
        result["dsr_status"] = input_status
        return result
    trial_values = np.asarray(trial_sharpes, dtype=float)
    if trial_values.ndim != 1 or not np.isfinite(trial_values).all():
        result["dsr_status"] = "TRIAL_SHARPE_DISTRIBUTION_INVALID"
        return result
    observed_sharpe = float(metrics["sharpe_ratio_per_trade"])
    expected_max_sharpe = _expected_maximum_sharpe(trial_values)
    variance_term = _dsr_variance_term(
        float(metrics["return_skewness"]),
        float(metrics["return_kurtosis_non_excess"]),
        observed_sharpe,
    )
    if (
        not math.isfinite(expected_max_sharpe)
        or not math.isfinite(variance_term)
        or variance_term <= 0.0
    ):
        result["dsr_status"] = "DSR_VARIANCE_INVALID"
        return result
    z_score = (
        (observed_sharpe - expected_max_sharpe)
        * math.sqrt(int(metrics["return_observation_count"]) - 1)
        / math.sqrt(variance_term)
    )
    probability = NormalDist().cdf(z_score)
    if not math.isfinite(z_score) or not math.isfinite(probability):
        result["dsr_status"] = "DSR_RESULT_NON_FINITE"
        return result
    result.update(
        {
            "dsr_z_score": float(z_score),
            "dsr_probability": probability,
            "dsr_expected_max_sharpe_per_trade": float(expected_max_sharpe),
            "dsr_status": "COMPUTED",
        }
    )
    return result


def _attach_dsr_to_cohort(metric_rows: list[dict[str, Any]], trial_scope: str) -> None:
    trial_sharpes = [
        float(metrics["sharpe_ratio_per_trade"])
        for metrics in metric_rows
        if metrics.get("sharpe_ratio_per_trade") is not None
        and math.isfinite(float(metrics["sharpe_ratio_per_trade"]))
    ]
    for metrics in metric_rows:
        metrics.update(
            _deflated_sharpe_diagnostics(metrics, trial_sharpes, trial_scope)
        )
        metrics["zai_net_pf_gt_1_1"] = _profit_factor(metrics) > 1.1
        z_score = metrics.get("dsr_z_score")
        metrics["zai_dsr_z_gt_1_0"] = z_score is not None and float(z_score) > 1.0


def _attach_dsr_diagnostics(
    candidates: list[dict[str, Any]], exit_matrix: dict[str, Any]
) -> None:
    candidate_metrics = [cast(dict[str, Any], item["metrics"]) for item in candidates]
    geometry_metrics = [cast(dict[str, Any], item) for item in exit_matrix["cells"]]
    _attach_dsr_to_cohort(candidate_metrics, "CURRENT_RUN_CANDIDATE_COHORT_ONLY")
    _attach_dsr_to_cohort(geometry_metrics, "CURRENT_RUN_EXIT_GEOMETRY_COHORT_ONLY")


def _candidate_passes(
    metrics: dict[str, Any],
    expectancy: float,
    primary_metrics: dict[str, Any] | None = None,
    primary_expectancy: float | None = None,
) -> bool:
    drawdown = metrics.get("max_drawdown_percent")
    win_rate = metrics.get("win_rate")
    passes = (
        int(metrics.get("trade_count", 0)) >= MINIMUM_TRADES
        and _profit_factor(metrics) > MINIMUM_NET_PF
        and drawdown is not None
        and float(drawdown) < MAXIMUM_DRAWDOWN
        and win_rate is not None
        and float(win_rate) > MINIMUM_WIN_RATE
    )
    if primary_metrics is None or primary_expectancy is None:
        return passes
    return (
        passes
        and _profit_factor(metrics) > _profit_factor(primary_metrics)
        and expectancy > primary_expectancy
    )


def _decision_rows(
    bars: DataFrame,
    event_indices: list[int],
    probabilities: dict[int, float] | None,
    threshold: float | None,
    mode: str,
    position_fractions: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in event_indices:
        probability = 1.0 if probabilities is None else probabilities.get(index)
        accepted = threshold is None or (
            probability is not None and probability >= threshold
        )
        rows.append(
            {
                "candidate_id": _candidate_id(mode),
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME,
                "mode": mode,
                "event_index": index,
                "signal_timestamp": pd.Timestamp(bars["date"].iloc[index]).isoformat(),
                "execution_timestamp": pd.Timestamp(
                    bars["date"].iloc[index + 1]
                ).isoformat(),
                "meta_probability": probability,
                "selected_meta_probability_threshold": threshold,
                "position_fraction": (
                    position_fractions.get(index)
                    if position_fractions is not None
                    else None
                ),
                "meta_filter_decision": "ACCEPT" if accepted else "REJECT",
                "executable_signal": "BUY" if accepted else "NO_ENTRY",
                "split_role": "FORWARD_VALIDATION",
            }
        )
    return rows


def _skipped_decision_rows(
    bars: DataFrame,
    event_indices: list[int],
    position_fractions: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": _candidate_id(META_CANDIDATE),
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "mode": META_CANDIDATE,
            "event_index": index,
            "signal_timestamp": pd.Timestamp(bars["date"].iloc[index]).isoformat(),
            "execution_timestamp": pd.Timestamp(
                bars["date"].iloc[index + 1]
            ).isoformat(),
            "meta_probability": None,
            "selected_meta_probability_threshold": None,
            "position_fraction": (
                position_fractions.get(index)
                if position_fractions is not None
                else None
            ),
            "meta_filter_decision": "SKIPPED",
            "executable_signal": "NOT_EVALUATED",
            "split_role": "FORWARD_VALIDATION",
        }
        for index in event_indices
    ]


def _accepted_candidate_events(
    event_indices: list[int],
    threshold: float | None,
    probabilities: dict[int, float] | None,
) -> list[int]:
    if threshold is None:
        return list(event_indices)
    if probabilities is None:
        return []
    return [
        index
        for index in event_indices
        if probabilities.get(index) is not None
        and float(probabilities[index]) >= threshold
    ]


def _candidate_summary(
    mode: str,
    event_count: int,
    accepted_count: int,
    threshold: float | None,
    position_fractions: dict[int, float] | None,
    metrics: dict[str, Any],
    expectancy: float,
    primary_metrics: dict[str, Any] | None,
    primary_expectancy: float | None,
) -> dict[str, Any]:
    return {
        "candidate_id": _candidate_id(mode),
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "mode": mode,
        "status": "COMPLETED",
        "filter": "ADX14_GT20_DIAGNOSTIC_ABLATION" if mode == ADX_ABLATION else "NONE",
        "primary_signal_count": event_count,
        "accepted_signal_count": accepted_count,
        "accepted_signal_fraction": accepted_count / event_count
        if event_count
        else 0.0,
        "selected_meta_probability_threshold": threshold,
        "position_sizing_method": (
            "INVERSE_24H_REALIZED_VOLATILITY"
            if position_fractions is not None
            else "FIXED_CONFIG"
        ),
        "metrics": metrics,
        "acceptance_target_met": _candidate_passes(
            metrics, expectancy, primary_metrics, primary_expectancy
        ),
        "sequence_model_used": False,
    }


def _enrich_candidate_trades(
    trades: list[dict[str, Any]], candidate_id: str, mode: str
) -> list[dict[str, Any]]:
    return [
        {
            **trade,
            "candidate_id": candidate_id,
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "mode": mode,
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "evaluation_split": "FORWARD_VALIDATION",
        }
        for trade in trades
    ]


def _run_candidate(
    bars: DataFrame,
    event_indices: list[int],
    forward_end: int,
    mode: str,
    threshold: float | None,
    probabilities: dict[int, float] | None,
    primary_metrics: dict[str, Any] | None = None,
    primary_expectancy: float | None = None,
    position_fractions: dict[int, float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = _accepted_candidate_events(event_indices, threshold, probabilities)
    actions = v4ac._build_forward_actions(bars, accepted, forward_end)
    if position_fractions is not None:
        _apply_position_fractions(actions, position_fractions)
    replay = simulate_signals(
        bars,
        actions,
        _execution_config(),
        trade_management_config=_trade_management_config(),
    )
    trades = cast(list[dict[str, Any]], replay["trades"])
    metrics = v4ac._summarize_replay(replay)
    metrics.update(_trade_risk_metrics(trades))
    expectancy = _expectancy(trades)
    metrics["mean_unscaled_net_expectancy_usd_at_full_equity"] = expectancy
    candidate = _candidate_summary(
        mode,
        len(event_indices),
        len(accepted),
        threshold,
        position_fractions,
        metrics,
        expectancy,
        primary_metrics,
        primary_expectancy,
    )
    enriched = _enrich_candidate_trades(trades, str(candidate["candidate_id"]), mode)
    decisions = _decision_rows(
        bars, event_indices, probabilities, threshold, mode, position_fractions
    )
    return candidate, enriched, decisions


def _skipped_meta_candidate(reason: str, event_count: int) -> dict[str, Any]:
    return {
        "candidate_id": _candidate_id(META_CANDIDATE),
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "mode": META_CANDIDATE,
        "status": "SKIPPED_INSUFFICIENT_META_SUPPORT",
        "skip_reason": reason,
        "filter": "NONE",
        "primary_signal_count": event_count,
        "accepted_signal_count": 0,
        "accepted_signal_fraction": 0.0,
        "selected_meta_probability_threshold": None,
        "metrics": {
            "trade_count": 0,
            "net_profit_factor": None,
            "net_profit_factor_infinite_no_losses": False,
            "max_drawdown_percent": None,
            "win_rate": None,
            "net_pnl_total_usd": 0.0,
            "mean_unscaled_net_expectancy_usd_at_full_equity": 0.0,
            **_trade_risk_metrics([]),
        },
        "acceptance_target_met": False,
        "sequence_model_used": False,
    }


def _run_exit_geometry_matrix(
    bars: DataFrame,
    event_indices: list[int],
    forward_end: int,
    position_fractions: dict[int, float] | None = None,
) -> dict[str, Any]:
    actions = v4ac._build_forward_actions(bars, event_indices, forward_end)
    if position_fractions is not None:
        _apply_position_fractions(actions, position_fractions)
    execution = _execution_config()
    cells: list[dict[str, Any]] = []
    for policy in EXIT_POLICIES:
        management = _trade_management_config(cast(dict[str, Any], policy))
        replay = simulate_signals(
            bars, actions, execution, trade_management_config=management
        )
        trades = cast(list[dict[str, Any]], replay["trades"])
        metrics = v4ac._summarize_replay(replay)
        cells.append(
            {
                "policy": policy["name"],
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME,
                "initial_stop_atr": policy["initial_stop_atr"],
                "take_profit_atr": policy["take_profit_atr"],
                "trailing_activation_atr": policy["trailing_activation_atr"],
                "trailing_step_atr": policy["trailing_step_atr"],
                "horizon_candles": int(PRIMARY_GEOMETRY["horizon_bars"]),
                "fee_bps_per_side": FEE_BPS_PER_SIDE,
                "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
                "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
                "trade_count": metrics["trade_count"],
                "win_rate": metrics["win_rate"],
                "net_profit_factor": metrics["net_profit_factor"],
                "net_profit_factor_infinite_no_losses": metrics[
                    "net_profit_factor_infinite_no_losses"
                ],
                "max_drawdown_percent": metrics["max_drawdown_percent"],
                "net_pnl_total_usd": metrics["net_pnl_total_usd"],
                "mean_unscaled_net_expectancy_usd_at_full_equity": _expectancy(trades),
                **_trade_risk_metrics(trades),
                "signal_count": len(event_indices),
                "evaluation_split": "FORWARD_VALIDATION",
                "meta_model_used": False,
                "final_holdout_used": False,
                "diagnostic_only": True,
            }
        )
    matrix = {
        "evaluation_split": "FORWARD_VALIDATION",
        "final_holdout_used": False,
        "meta_model_used": False,
        "promotion_allowed": False,
        "cost_contract": {
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        },
        "policies": [dict(item) for item in EXIT_POLICIES],
        "cells": cells,
    }
    _validate_exit_matrix(matrix)
    return matrix


def _validate_exit_matrix_scope(matrix: dict[str, Any]) -> None:
    valid = (
        matrix.get("evaluation_split") == "FORWARD_VALIDATION"
        and matrix.get("final_holdout_used") is False
        and matrix.get("meta_model_used") is False
        and matrix.get("promotion_allowed") is False
    )
    if not valid:
        raise ValueError("v4ai_exit_matrix_scope_invalid")


def _validate_exit_matrix_cost_contract(matrix: dict[str, Any]) -> None:
    expected = {
        "fee_bps_per_side": FEE_BPS_PER_SIDE,
        "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
    }
    if matrix.get("cost_contract") != expected:
        raise ValueError("v4ai_exit_matrix_cost_invalid")


def _exit_matrix_cells(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    cells = matrix.get("cells")
    if not isinstance(cells, list) or len(cells) != len(EXIT_POLICIES):
        raise ValueError("v4ai_exit_matrix_incomplete")
    return cast(list[dict[str, Any]], cells)


def _validate_exit_matrix_policy_set(cells: list[dict[str, Any]]) -> None:
    expected = {item["name"] for item in EXIT_POLICIES}
    actual = {item.get("policy") for item in cells}
    if actual != expected:
        raise ValueError("v4ai_exit_matrix_policy_invalid")


def _validate_exit_matrix_cell_cost(cell: dict[str, Any]) -> None:
    valid = (
        cell.get("round_trip_cost_bps") == ROUND_TRIP_COST_BPS
        and cell.get("fee_bps_per_side") == FEE_BPS_PER_SIDE
        and cell.get("slippage_bps_per_side") == SLIPPAGE_BPS_PER_SIDE
    )
    if not valid:
        raise ValueError("v4ai_exit_matrix_cell_invalid")


def _validate_exit_matrix_cell_market(cell: dict[str, Any]) -> None:
    if (
        cell.get("symbol") != SYMBOL
        or cell.get("timeframe") != TIMEFRAME
        or cell.get("horizon_candles") != int(PRIMARY_GEOMETRY["horizon_bars"])
    ):
        raise ValueError("v4ai_exit_matrix_cell_invalid")


def _validate_exit_matrix_cell_scope(cell: dict[str, Any]) -> None:
    valid = (
        cell.get("diagnostic_only") is True
        and cell.get("meta_model_used") is False
        and cell.get("final_holdout_used") is False
    )
    if not valid:
        raise ValueError("v4ai_exit_matrix_cell_invalid")


def _validate_exit_matrix_cell_geometry(cell: dict[str, Any]) -> None:
    policy_name = cell.get("policy")
    policy = next(
        (item for item in EXIT_POLICIES if item["name"] == policy_name),
        None,
    )
    if policy is None:
        raise ValueError("v4ai_exit_matrix_policy_invalid")
    for field in (
        "initial_stop_atr",
        "take_profit_atr",
        "trailing_activation_atr",
        "trailing_step_atr",
    ):
        if cell.get(field) != policy[field]:
            raise ValueError("v4ai_exit_matrix_geometry_invalid")


def _validate_exit_matrix_cell(cell: dict[str, Any]) -> None:
    _validate_exit_matrix_cell_cost(cell)
    _validate_exit_matrix_cell_market(cell)
    _validate_exit_matrix_cell_scope(cell)
    _validate_exit_matrix_cell_geometry(cell)


def _validate_exit_matrix(matrix: dict[str, Any]) -> None:
    _validate_exit_matrix_scope(matrix)
    _validate_exit_matrix_cost_contract(matrix)
    cells = _exit_matrix_cells(matrix)
    _validate_exit_matrix_policy_set(cells)
    for cell in cells:
        _validate_exit_matrix_cell(cell)


def _risk_metric_contract() -> dict[str, Any]:
    return {
        "return_basis": "NET_CLOSED_TRADE_RETURN_FRACTION",
        "observation_unit": "CLOSED_TRADE",
        "annualization": "NONE_IRREGULAR_TRADE_INTERVALS",
        "sortino_target_return_per_trade": SORTINO_TARGET_RETURN_PER_TRADE,
        "dsr_method": "BAILEY_LOPEZ_DE_PRADO_2014",
        "dsr_probability_definition": "STANDARD_NORMAL_CDF_OF_DSR_Z_SCORE",
        "dsr_trial_cohorts": [
            "CURRENT_RUN_CANDIDATE_COHORT_ONLY",
            "CURRENT_RUN_EXIT_GEOMETRY_COHORT_ONLY",
        ],
        "historical_trials_included": False,
        "trial_independence_assumption": "UNVERIFIED",
    }


def _safety_report() -> dict[str, Any]:
    return {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "orders_or_execution_api_used": False,
        "credentials_used": False,
        "sequence_model_used": False,
    }


def _validate_report_identity(report: dict[str, Any], commit_sha: str) -> None:
    identity_valid = (
        report.get("schema_version") == SCHEMA
        and report.get("commit_sha") == commit_sha
    )
    if not identity_valid:
        raise ValueError("v4ai_report_identity_invalid")
    scope_valid = report.get("symbols") == [SYMBOL] and report.get("timeframes") == [
        TIMEFRAME
    ]
    if not scope_valid:
        raise ValueError("v4ai_report_scope_invalid")


def _validate_report_cost_contract(report: dict[str, Any]) -> None:
    expected = {
        "fee_type_assumption": "MAKER_FEE_PROXY",
        "fee_bps_per_side": FEE_BPS_PER_SIDE,
        "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "passive_limit_fill_simulation": False,
    }
    if report.get("cost_contract") != expected:
        raise ValueError("v4ai_report_cost_contract_invalid")


def _validate_report_risk_contract(report: dict[str, Any]) -> None:
    if report.get("risk_metric_contract") != _risk_metric_contract():
        raise ValueError("v4ai_risk_metric_contract_invalid")


def _validate_report_primary_identity(
    report: dict[str, Any], diagnostics: dict[str, Any]
) -> None:
    if (
        report.get("primary_signal") != PRIMARY_SIGNAL_GENERATION
        or report.get("primary_filter") != PRIMARY_FILTER_CONTRACT
        or diagnostics.get("generation") != PRIMARY_SIGNAL_GENERATION
        or diagnostics.get("lookahead_used") is not False
    ):
        raise ValueError("v4ai_report_primary_filter_invalid")


def _validate_report_hmm_feature_parameters(
    diagnostics: dict[str, Any],
) -> None:
    hmm_features = diagnostics.get("hmm_features", {})
    valid = (
        hmm_features.get("generation") == HMM_FEATURE_GENERATION
        and hmm_features.get("realized_volatility_window_bars")
        == HMM_FEATURE_WINDOW_BARS
        and hmm_features.get("lookahead_used") is False
        and hmm_features.get("cusum_used") is False
        and hmm_features.get("final_holdout_features_unavailable") is True
        and "cusum" not in diagnostics
    )
    if not valid:
        raise ValueError("v4an_report_hmm_feature_contract_invalid")


def _validate_report_selector_parameters(diagnostics: dict[str, Any]) -> None:
    selector = diagnostics.get("selector", {})
    valid = (
        selector.get("selector_type") == "HMM_REGIME_HARD_SELECTOR"
        and selector.get("hard_selector_enabled") is True
        and selector.get("soft_blend_enabled") is False
        and selector.get("trend_ema_period") == EMA_FAST_PERIOD
        and selector.get("range_rsi_period") == RSI_PERIOD
        and selector.get("range_rsi_threshold") == RANGE_RSI_ENTRY_THRESHOLD
        and selector.get("momentum_strategy")
        == "CLOSE_ABOVE_EMA50_TRENDING_UP_ONLY_SPOT_LONG"
        and selector.get("mean_reversion_strategy") == "WILDER_RSI14_LT45_RANGING_ONLY"
        and selector.get("downtrend_policy") == "RISK_OFF_NO_LONG_ENTRY_SPOT_ONLY"
        and selector.get("no_cusum_filter") is True
        and selector.get("short_entries_enabled") is False
        and selector.get("lookahead_used") is False
    )
    if not valid:
        raise ValueError("v4ai_report_regime_selector_invalid")


def _validate_report_sizing_parameters(diagnostics: dict[str, Any]) -> None:
    sizing = diagnostics.get("volatility_targeting", {})
    valid = (
        sizing.get("method") == "INVERSE_24H_REALIZED_VOLATILITY"
        and sizing.get("target_reference")
        == "MEDIAN_OF_OUTER_TRAIN_48_BAR_REALIZED_VOLATILITY"
        and sizing.get("sizing_cap") == 1.0
        and sizing.get("leverage_enabled") is False
        and sizing.get("training_rows", 0) >= regime_engine.HMM_MINIMUM_TRAIN_ROWS
    )
    if not valid:
        raise ValueError("v4ai_report_volatility_target_contract_invalid")


def _validate_report_primary_filter(report: dict[str, Any]) -> None:
    diagnostics = report.get("primary_signal_diagnostics", {})
    _validate_report_primary_identity(report, diagnostics)
    _validate_report_hmm_feature_parameters(diagnostics)
    _validate_report_selector_parameters(diagnostics)
    _validate_report_sizing_parameters(diagnostics)
    if diagnostics.get("research_signal_count_target") != RESEARCH_SIGNAL_COUNT_TARGET:
        raise ValueError("v4ai_report_primary_filter_invalid")


def _validate_regime_model_identity(regime: dict[str, Any]) -> None:
    valid = (
        regime.get("algorithm") == "GAUSSIAN_HMM_DIAGONAL"
        and regime.get("state_count") == regime_engine.HMM_STATE_COUNT
        and regime.get("fit_scope") == "OUTER_TRAIN_ONLY"
        and regime.get("estimation_method") == "BAUM_WELCH_TRAINING_SLICE_ONLY"
        and regime.get("posterior_method") == "CAUSAL_FORWARD_FILTER"
        and regime.get("smoothing_used") is False
        and regime.get("lookahead_used") is False
        and regime.get("fit_rows", 0) >= regime_engine.HMM_MINIMUM_TRAIN_ROWS
        and regime.get("final_holdout_features_unavailable") is True
    )
    if not valid:
        raise ValueError("v4ai_report_hmm_causality_contract_invalid")


def _validate_regime_chronological_bounds(
    regime: dict[str, Any], chronological: dict[str, Any]
) -> None:
    train_end = chronological.get("train_end_index_exclusive")
    forward_end = chronological.get("forward_validation_end_index_exclusive")
    if (
        regime.get("fit_end_index_exclusive") != train_end
        or regime.get("apply_end_index_exclusive") != forward_end
    ):
        raise ValueError("v4ai_report_hmm_causality_contract_invalid")


def _validate_report_regime(report: dict[str, Any]) -> None:
    regime = report.get("regime_diagnostics", {})
    chronological = report.get("chronological_evaluation", {})
    _validate_regime_model_identity(regime)
    _validate_regime_chronological_bounds(regime, chronological)


def _validate_report_geometry(report: dict[str, Any]) -> None:
    if report.get("fixed_exit_geometry") != PRIMARY_GEOMETRY:
        raise ValueError("v4ai_report_fixed_exit_geometry_invalid")


def _validate_report_holdout(report: dict[str, Any]) -> None:
    if report.get("final_holdout_used") is not False:
        raise ValueError("v4ai_final_holdout_contamination")
    chronological = report.get("chronological_evaluation", {})
    if chronological.get("final_15_percent_holdout_untouched") is not True:
        raise ValueError("v4ai_final_holdout_not_untouched")


def _validate_report_candidates(report: dict[str, Any]) -> None:
    candidates = report.get("candidates", [])
    modes = [item.get("mode") for item in candidates]
    if modes != list(CANDIDATE_NAMES):
        raise ValueError("v4ai_candidate_matrix_invalid")
    for item in candidates:
        if item.get("sequence_model_used") is not False:
            raise ValueError("v4ai_sequence_model_forbidden")


def _validate_report_safety(report: dict[str, Any]) -> None:
    safety = report.get("safety", {})
    if safety.get("Champion") is not None:
        raise ValueError("v4ai_fail_closed_safety_contract_invalid")
    if safety.get("order_permission") is not False:
        raise ValueError("v4ai_fail_closed_safety_contract_invalid")
    if safety.get("trading_authorization") != "NO-GO":
        raise ValueError("v4ai_fail_closed_safety_contract_invalid")
    if safety.get("private_api_calls") != 0 or safety.get("real_order_attempts") != 0:
        raise ValueError("v4ai_fail_closed_safety_contract_invalid")
    if safety.get("sequence_model_used") is not False:
        raise ValueError("v4ai_fail_closed_safety_contract_invalid")


def _validate_meta_threshold(meta: dict[str, Any]) -> None:
    threshold = meta.get("threshold_diagnostics", {})
    valid = (
        threshold.get("status") == "SELECTED_TOP_70_PERCENT_ACCEPTANCE"
        and threshold.get("target_accept_fraction") == META_TARGET_ACCEPT_FRACTION
        and threshold.get("labels_used") is False
        and threshold.get("selection_scope") == "THIRD_PURGED_OOF_TRAINING_SLICE_ONLY"
    )
    if not valid:
        raise ValueError("v4ai_meta_threshold_or_separation_contract_invalid")


def _validate_meta_separation(meta: dict[str, Any]) -> None:
    separation = meta.get("separation_diagnostics", {})
    temperature = meta.get("temperature_diagnostics", {})
    valid_scope = (
        separation.get("metric_scope")
        == "PURGED_OOF_TEMPERATURE_SLICE_OUT_OF_FIT_PREDICTIONS"
    )
    if not valid_scope:
        raise ValueError("v4ai_meta_threshold_or_separation_contract_invalid")
    if separation.get("sample_count") != temperature.get("calibration_rows"):
        raise ValueError("v4ai_meta_threshold_or_separation_contract_invalid")
    roc_auc = float(separation.get("roc_auc", -1.0))
    average_precision = float(separation.get("pr_auc_average_precision", -1.0))
    if not 0.0 <= roc_auc <= 1.0 or not 0.0 <= average_precision <= 1.0:
        raise ValueError("v4ai_meta_threshold_or_separation_contract_invalid")


def _validate_meta_report(meta: dict[str, Any]) -> None:
    separation = meta.get("separation_diagnostics", {})
    if meta.get("enabled"):
        _validate_meta_threshold(meta)
        _validate_meta_separation(meta)
        return
    if separation.get("status") != "NOT_AVAILABLE_MODEL_NOT_FIT":
        raise ValueError("v4ai_separation_status_missing")
    skip_reason = meta.get("skip_reason")
    if not isinstance(skip_reason, str) or not skip_reason:
        raise ValueError("v4ai_meta_skip_reason_missing")
    if not _support_shortfall(ValueError(skip_reason)):
        raise ValueError("v4ai_meta_skip_reason_not_supported")


def _validate_report(report: dict[str, Any], commit_sha: str) -> None:
    _validate_report_identity(report, commit_sha)
    _validate_report_cost_contract(report)
    _validate_report_risk_contract(report)
    _validate_report_primary_filter(report)
    _validate_report_regime(report)
    _validate_report_geometry(report)
    _validate_report_holdout(report)
    _validate_report_candidates(report)
    _validate_exit_matrix(report.get("exit_geometry_matrix", {}))
    _validate_report_safety(report)
    _validate_meta_report(report.get("meta_model", {}))


def _format_optional_ratio(value: Any) -> str:
    if value is None:
        return "N/A"
    numeric = float(value)
    return f"{numeric:.4f}" if math.isfinite(numeric) else "N/A"


def _format_diagnostic_flag(value: Any) -> str:
    if value is None:
        return "N/A"
    return "YES" if bool(value) else "NO"


def _format_pf(metrics: dict[str, Any]) -> str:
    if metrics.get("net_profit_factor_infinite_no_losses"):
        return "INF"
    value = metrics.get("net_profit_factor")
    return "N/A" if value is None else f"{float(value):.4f}"


def _format_optional_percent(value: Any, scale: float = 1.0) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * scale:.2f}"


def _format_optional_fixed(value: Any, precision: int) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{precision}f}"


def _format_acceptance(item: dict[str, Any]) -> str:
    return "PASS" if item.get("acceptance_target_met") else "NO"


def _render_step_summary_header(report: dict[str, Any]) -> list[str]:
    return [
        "## AliX V4-AN ETHUSDT 30m HMM Regime-Adaptive Economic Validation",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "Scope: ETHUSDT Spot, 30m analysis bars built from provenance-verified 1m public OHLCV; 1825 days requested with trim to latest contiguous segment.",
        "Primary event trigger: every completed 30m candle is evaluated; no CUSUM event or CUSUM-derived filter is used.",
        "Regime model: diagonal Gaussian HMM(3), fit on outer-train only and forward-filtered causally, without smoothing.",
        "HMM input volatility: trailing 48-bar realized volatility (24 hours on 30m bars).",
        "Hard selector: dominant HMM posterior only; Trending Up buys on Close > EMA50, Ranging buys on Wilder RSI(14)<45, Trending Down remains flat.",
        "No soft strategy blend, no short entries, no pyramiding, and no leverage; the economic simulator remains Spot-long only.",
        "Execution: closed 30m candle signal, next-bar-open entry.",
        "ADX20 remains diagnostic-ablation only and does not gate the Primary candidate.",
        "Volatility targeting: inverse 24-hour realized volatility from the 48-bar feature, using the outer-train median and a 1x cap.",
        "Maker-cost assumption: 10 bps fee + 5 bps slippage per side = 30 bps round trip.",
        "Important: fixed maker-fee proxy only; passive Limit queue/partial-fill mechanics are not simulated.",
        "Risk returns: net closed-trade returns; Sharpe and Sortino are per-trade, not calendar-annualized; Sortino target=0.",
        "DSR: Bailey-López de Prado expected-maximum-Sharpe deflation; DSR Z is unbounded and DSR probability=Phi(Z) is in [0, 1].",
        "DSR trials are limited to each current-run comparison cohort; historical trials are not included and trial independence is unverified.",
        "Z.AI diagnostic reference (not a promotion gate): Net PF >1.1 and DSR Z >1.0; the existing Net PF >1.5 acceptance gate remains unchanged.",
        "Fixed candidate geometry: SL=1.5 ATR, TP=2.5 ATR, activation=1.5 ATR, step=2.0 ATR, horizon=48 bars.",
        "Training: outer 70%; purged OOF Fit/Temperature/Threshold slices=70/15/15.",
        "Evaluation: next 15% forward validation; final 15% holdout untouched.",
        "Meta: LightGBM + TreeSHAP top-30 Fit-only selection; causal HMM posterior probabilities, 48-bar realized volatility, EMA and RSI features are used; no CUSUM feature, no LSTM/sequence model.",
        "Meta cutoff: accept the upper 70% of probabilities using the 30th percentile order statistic on the third purged OOF slice; ties are included.",
    ]


def _render_candidate_row(item: dict[str, Any]) -> str:
    metrics = item["metrics"]
    return (
        f"| {item['mode']} | {item.get('primary_signal_count', 0)} "
        f"| {item.get('accepted_signal_count', 0)} "
        f"| {metrics.get('trade_count', 0)} "
        f"| {_format_optional_percent(metrics.get('win_rate'), 100.0)} "
        f"| {_format_pf(metrics)} "
        f"| {_format_optional_percent(metrics.get('max_drawdown_percent'))} "
        f"| {_format_optional_ratio(metrics.get('dsr_z_score'))} "
        f"| {_format_optional_ratio(metrics.get('dsr_probability'))} "
        f"| {metrics.get('dsr_trial_count', 'N/A')} "
        f"| {_format_optional_ratio(metrics.get('sortino_ratio_per_trade'))} "
        f"| {_format_diagnostic_flag(metrics.get('zai_net_pf_gt_1_1'))} "
        f"| {_format_diagnostic_flag(metrics.get('zai_dsr_z_gt_1_0'))} "
        f"| {_format_optional_fixed(metrics.get('net_pnl_total_usd') or 0.0, 2)} "
        f"| {_format_optional_fixed(metrics.get('mean_unscaled_net_expectancy_usd_at_full_equity'), 4)} "
        f"| {_format_acceptance(item)} "
        f"| {item['status']} |"
    )


def _render_candidate_section(candidates: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Candidate | Signals | Accepted | Trades | Win Rate % | Net PF | Max DD % | DSR Z | DSR P | DSR N | Sortino/trade | Z.AI PF>1.1 | Z.AI DSR Z>1 | Net PnL USD | Net Expectancy USD/trade | Acceptance | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|",
    ]
    lines.extend(_render_candidate_row(item) for item in candidates)
    return lines


def _render_meta_diagnostics(report: dict[str, Any]) -> list[str]:
    meta = report.get("meta_model", {})
    training = meta.get("training_diagnostics", {})
    split = meta.get("split_diagnostics", {})
    separation = meta.get("separation_diagnostics", {})
    fit = meta.get("fit_diagnostics", {})
    threshold = meta.get("threshold_diagnostics", {})
    label_counts = training.get("label_class_counts", {})
    fit_counts = split.get("fit_class_counts", {})
    temperature_counts = split.get("temperature_class_counts", {})
    threshold_counts = split.get("threshold_class_counts", {})
    primary = report.get("primary_signal_diagnostics", {})
    hmm_features = primary.get("hmm_features", {})
    selector = primary.get("selector", {})
    sizing = primary.get("volatility_targeting", {})
    regime = report.get("regime_diagnostics", {})
    return [
        "",
        (
            f"Dataset provenance: requested_days={report['dataset_provenance'].get('requested_days', 'N/A')}; "
            f"1m_rows={report['dataset_provenance'].get('row_count', 'N/A')}; "
            f"missing_minutes={report['dataset_provenance'].get('missing_minute_intervals', 0)}; "
            f"30m_bars_after_trim={report.get('bar_count', 0)}; "
            f"trim={report['dataset_provenance'].get('resample_gap_trim', {}).get('status', 'N/A')}; "
            f"trimmed_bars={report['dataset_provenance'].get('resample_gap_trim', {}).get('trimmed_bars', 0)}; "
            f"contiguous_start={report['dataset_provenance'].get('resample_gap_trim', {}).get('segment_start_utc', 'N/A')}."
        ),
        (
            f"HMM input volatility window={hmm_features.get('realized_volatility_window_bars', 'N/A')} bars; "
            f"valid volatility rows={hmm_features.get('valid_realized_volatility_rows', 0)}; "
            f"lookahead={hmm_features.get('lookahead_used', 'N/A')}; "
            f"CUSUM used={hmm_features.get('cusum_used', 'N/A')}."
        ),
        (
            f"Hard selector: Close>EMA50 signals={selector.get('momentum_signal_count', 0)}; "
            f"RSI<45 ranging signals={selector.get('mean_reversion_signal_count', 0)}; "
            f"dominant regimes Up/Down-risk-off/Range="
            f"{selector.get('dominant_trending_up_rows', 0)}/"
            f"{selector.get('dominant_trending_down_risk_off_rows', 0)}/"
            f"{selector.get('dominant_ranging_rows', 0)}; "
            f"BUY={selector.get('primary_buy_count', 0)}; hard-only={selector.get('hard_selector_enabled', False)}."
        ),
        (
            "Primary BUYs research/train/forward: "
            f"{report['signal_counts']['primary_buy_count']}/"
            f"{report['signal_counts']['train_primary_buy_count']}/"
            f"{report['signal_counts']['forward_validation_primary_buy_count']}; "
            f"research target >=100="
            f"{'PASS' if report['signal_counts']['research_signal_target_met'] else 'WARNING'}; "
            f"Meta outer-train minimum="
            f"{report['signal_counts']['minimum_primary_train_signal_count']}."
        ),
        f"ADX20 diagnostic-ablation BUYs: {report['signal_counts']['adx_ablation_buy_count']}.",
        (
            f"Gaussian HMM: states={regime.get('state_count', 'N/A')}; "
            f"fit_rows={regime.get('fit_rows', 0)}; fit_end={regime.get('fit_end_index_exclusive', 'N/A')}; "
            f"fit_scope={regime.get('fit_scope', 'N/A')}; "
            f"posterior={regime.get('posterior_method', 'N/A')}; "
            f"smoothing={regime.get('smoothing_used', 'N/A')}; "
            f"lookahead={regime.get('lookahead_used', 'N/A')}."
        ),
        (
            f"Volatility targeting: method={sizing.get('method', 'N/A')}; "
            f"training median target={sizing.get('target_volatility', 'N/A')}; "
            f"position fraction min/median/max={sizing.get('fraction_min', 'N/A')}/"
            f"{sizing.get('fraction_median', 'N/A')}/{sizing.get('fraction_max', 'N/A')}; "
            f"leverage={sizing.get('leverage_enabled', 'N/A')}."
        ),
        f"OOF primary events: {training.get('oof_primary_event_count', 0)}; "
        f"matured costed OOF labels: {training.get('matured_costed_oof_label_count', 0)}; "
        f"immature excluded: {training.get('immature_oof_event_count', 0)}.",
        (
            "OOF label classes failure/success: "
            f"{label_counts.get('meta_failure', 'N/A')}/"
            f"{label_counts.get('meta_success', 'N/A')}."
        ),
        (
            "Purged OOF slices rows fit/temperature/threshold: "
            f"{split.get('fit_rows', 'N/A')}/"
            f"{split.get('temperature_rows', 'N/A')}/"
            f"{split.get('threshold_rows', 'N/A')}."
        ),
        (
            "Purged OOF classes failure/success fit="
            f"{fit_counts.get('meta_failure', 'N/A')}/"
            f"{fit_counts.get('meta_success', 'N/A')}; temperature="
            f"{temperature_counts.get('meta_failure', 'N/A')}/"
            f"{temperature_counts.get('meta_success', 'N/A')}; threshold="
            f"{threshold_counts.get('meta_failure', 'N/A')}/"
            f"{threshold_counts.get('meta_success', 'N/A')}."
        ),
        f"Meta fit: {'ENABLED' if meta.get('enabled') else 'SKIPPED'}; "
        f"calibration={split.get('temperature_calibration_mode', 'N/A')}; "
        f"reason={meta.get('skip_reason') or 'N/A'}.",
        (
            f"WARNING: Meta was skipped ({meta.get('skip_reason')}); Primary metrics remain reportable, "
            "Meta validation is unavailable, and promotion remains blocked."
        )
        if not meta.get("enabled")
        else "Meta model trained; forward validation is reported separately.",
        f"SHAP top-k: {fit.get('top_k_selected', 0)}.",
        (
            f"Separation: ROC-AUC={separation.get('roc_auc', 'N/A')}; "
            f"PR-AUC/AP={separation.get('pr_auc_average_precision', 'N/A')}; "
            f"positives={separation.get('positive_count', 'N/A')}/"
            f"{separation.get('sample_count', 'N/A')}; "
            f"support={separation.get('positive_support_quality', 'N/A')}."
        ),
        (
            f"Meta target acceptance: {threshold.get('target_accept_fraction', 'N/A')}; "
            f"cutoff percentile={threshold.get('cutoff_percentile', 'N/A')}; "
            f"threshold={threshold.get('threshold', 'N/A')}; "
            f"OOF threshold rows={threshold.get('threshold_rows', 0)}; "
            f"selected OOF fraction={threshold.get('selected_fraction', 'N/A')}."
        ),
    ]


def _render_exit_geometry_row(cell: dict[str, Any]) -> str:
    return (
        f"| {cell['policy']} | {float(cell['initial_stop_atr']):.1f} "
        f"| {float(cell['take_profit_atr']):.1f} "
        f"| {float(cell['trailing_activation_atr']):.1f} "
        f"| {float(cell['trailing_step_atr']):.1f} "
        f"| {float(cell['round_trip_cost_bps']):.0f} "
        f"| {cell['trade_count']} "
        f"| {_format_optional_percent(cell['win_rate'], 100.0)} "
        f"| {_format_pf(cell)} "
        f"| {_format_optional_percent(cell['max_drawdown_percent'])} "
        f"| {_format_optional_ratio(cell.get('dsr_z_score'))} "
        f"| {_format_optional_ratio(cell.get('dsr_probability'))} "
        f"| {cell.get('dsr_trial_count', 'N/A')} "
        f"| {_format_optional_ratio(cell.get('sortino_ratio_per_trade'))} "
        f"| {_format_diagnostic_flag(cell.get('zai_net_pf_gt_1_1'))} "
        f"| {_format_diagnostic_flag(cell.get('zai_dsr_z_gt_1_0'))} "
        f"| {_format_optional_fixed(cell['net_pnl_total_usd'], 2)} "
        f"| {_format_optional_fixed(cell['mean_unscaled_net_expectancy_usd_at_full_equity'], 4)} |"
    )


def _render_exit_geometry_section(cells: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "### Base Primary exit-geometry matrix (30 bps only)",
        "| Policy | SL ATR | TP ATR | Activation ATR | Step ATR | Cost bps RT | Trades | Win % | Net PF | Max DD % | DSR Z | DSR P | DSR N | Sortino/trade | Z.AI PF>1.1 | Z.AI DSR Z>1 | Net PnL USD | Expectancy USD/trade |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    lines.extend(_render_exit_geometry_row(cell) for cell in cells)
    return lines


def _render_step_summary_footer() -> list[str]:
    return [
        "",
        "Acceptance: Net PF >1.5, Max DD <15%, Win Rate >45%, Trades >=30; Meta must also outperform Base Primary.",
        "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0, sequence_model_used=false.",
        "Research only. No private exchange API, credentials, live/paper orders, or production execution.",
        "Promotion: BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW.",
    ]


def _render_step_summary(report: dict[str, Any]) -> str:
    if report.get("schema_version") != SCHEMA:
        raise ValueError("v4ai_step_summary_schema_invalid")
    lines = _render_step_summary_header(report)
    lines.extend(_render_candidate_section(report["candidates"]))
    lines.extend(_render_meta_diagnostics(report))
    lines.extend(_render_exit_geometry_section(report["exit_geometry_matrix"]["cells"]))
    lines.extend(_render_step_summary_footer())
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    if not columns:
        columns = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_report(
    commit_sha: str,
    provenance: dict[str, Any],
    bars: DataFrame,
    train_end: int,
    forward_end: int,
    signal_counts: dict[str, int],
    primary_signal_diagnostics: dict[str, Any],
    adx_ablation_diagnostics: dict[str, Any],
    train_gate: dict[str, Any],
    meta: dict[str, Any],
    regime_diagnostics: dict[str, Any],
    candidates: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    exit_matrix: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA,
        "commit_sha": commit_sha,
        "symbols": [SYMBOL],
        "timeframes": [TIMEFRAME],
        "dataset_days": DAYS,
        "dataset_provenance": provenance,
        "bar_count": int(len(bars)),
        "primary_signal": PRIMARY_SIGNAL_GENERATION,
        "primary_filter": dict(PRIMARY_FILTER_CONTRACT),
        "primary_signal_diagnostics": dict(primary_signal_diagnostics),
        "fixed_exit_geometry": dict(PRIMARY_GEOMETRY),
        "cost_contract": {
            "fee_type_assumption": "MAKER_FEE_PROXY",
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "passive_limit_fill_simulation": False,
        },
        "risk_metric_contract": _risk_metric_contract(),
        "signal_counts": signal_counts,
        "adx_ablation_diagnostics": adx_ablation_diagnostics,
        "primary_train_sample_gate": train_gate,
        "chronological_evaluation": {
            "train_end_index_exclusive": train_end,
            "forward_validation_end_index_exclusive": forward_end,
            "final_holdout_start_index": forward_end,
            "final_15_percent_holdout_untouched": True,
        },
        "final_holdout_used": False,
        "regime_diagnostics": regime_diagnostics,
        "meta_model": {
            "enabled": "model" in meta,
            **{key: value for key, value in meta.items() if key != "model"},
        },
        "candidates": candidates,
        "exit_geometry_matrix": exit_matrix,
        "trade_count": len(trades),
        "decision_count": len(decisions),
        "safety": _safety_report(),
        "promotion_status": "BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW",
    }
    _validate_report(report, commit_sha)
    return report


def _empty_meta(reason: str) -> dict[str, Any]:
    return {
        "skip_reason": reason,
        "training_diagnostics": {},
        "split_diagnostics": {},
        "fit_diagnostics": {},
        "shap_rankings": [],
        "temperature_diagnostics": {},
        "separation_diagnostics": {"status": "NOT_AVAILABLE_MODEL_NOT_FIT"},
        "threshold_diagnostics": {"status": "NOT_AVAILABLE_MODEL_NOT_FIT"},
    }


def run_v4ai(
    input_dir: Path,
    output_report: Path,
    output_trades_csv: Path,
    output_decisions_csv: Path,
    *,
    days: int,
    commit_sha: str,
) -> dict[str, Any]:
    _validate_request(days, commit_sha)
    (
        provenance,
        bars,
        train_end,
        forward_end,
        research_bars,
        base_features,
        _legacy_breakout_signal,
    ) = v4af._load_research_context(
        input_dir,
        days,
        commit_sha,
        allow_exchange_gaps=ALLOW_EXCHANGE_GAPS,
        timeframe=TIMEFRAME,
    )
    technical_features, _legacy_static_signal, _technical_diagnostics = (
        _ema_trend_rsi_primary_features(research_bars, base_features)
    )
    features, hmm_feature_diagnostics = _hmm_feature_inputs(
        research_bars,
        technical_features,
        apply_end=forward_end,
    )
    features, regime_diagnostics = _regime_features(
        features, train_end=train_end, apply_end=forward_end
    )
    features, primary_signal, selector_diagnostics = _regime_reactive_primary_signal(
        research_bars, features, apply_end=forward_end
    )
    adx_ablation_signal, adx_ablation_diagnostics = _apply_adx_ablation(
        primary_signal, features
    )
    horizon_candles = int(PRIMARY_GEOMETRY["horizon_bars"])
    primary_window_counts, train_gate, primary_event_indices = (
        v4af._primary_signal_evaluation(
            primary_signal,
            train_end,
            forward_end,
            horizon_candles=horizon_candles,
            train_start_index=REGIME_WARMUP_BARS,
        )
    )
    adx_ablation_event_indices = v4af._eligible_primary_event_indices(
        adx_ablation_signal,
        train_end,
        forward_end,
        horizon_candles=horizon_candles,
    )
    position_fractions, sizing_diagnostics = _volatility_target_position_fractions(
        features, train_end=train_end, event_indices=primary_event_indices
    )
    primary_signal_diagnostics = {
        "generation": PRIMARY_SIGNAL_GENERATION,
        "hmm_features": hmm_feature_diagnostics,
        "selector": selector_diagnostics,
        "volatility_targeting": sizing_diagnostics,
        "research_signal_count_target": RESEARCH_SIGNAL_COUNT_TARGET,
        "lookahead_used": False,
        "execution_timing": "CLOSED_30M_BAR_NEXT_BAR_OPEN",
    }
    signal_counts = {
        "primary_buy_count": int(primary_signal.sum()),
        "train_primary_buy_count": int(
            primary_window_counts["train_primary_buy_count"]
        ),
        "forward_validation_primary_buy_count": int(
            primary_window_counts["forward_validation_primary_buy_count"]
        ),
        "minimum_primary_train_signal_count": int(
            train_gate["minimum_primary_buy_count"]
        ),
        "research_signal_target_met": int(primary_signal.sum())
        >= RESEARCH_SIGNAL_COUNT_TARGET,
        "adx_ablation_buy_count": int(adx_ablation_signal.sum()),
        "primary_forward_event_count": len(primary_event_indices),
        "adx_ablation_forward_event_count": len(adx_ablation_event_indices),
    }
    primary_candidate, primary_trades, primary_decisions = _run_candidate(
        research_bars,
        primary_event_indices,
        forward_end,
        BASE_CONTROL,
        None,
        None,
        position_fractions=position_fractions,
    )
    adx_ablation_candidate, adx_ablation_trades, adx_ablation_decisions = (
        _run_candidate(
            research_bars,
            adx_ablation_event_indices,
            forward_end,
            ADX_ABLATION,
            None,
            None,
            position_fractions=position_fractions,
        )
    )
    try:
        meta = _fit_meta_model(
            research_bars, features, primary_signal, train_end, train_gate
        )
    except ValueError as error:
        if not _support_shortfall(error):
            raise
        meta = _empty_meta(str(error))
    if meta.get("skip_reason") is None:
        probabilities = _forward_probability_map(
            features,
            primary_event_indices,
            cast(list[str], meta["selected_features"]),
            meta["model"],
            float(meta["temperature"]),
        )
        meta_candidate, meta_trades, meta_decisions = _run_candidate(
            research_bars,
            primary_event_indices,
            forward_end,
            META_CANDIDATE,
            float(meta["threshold"]),
            probabilities,
            primary_candidate["metrics"],
            float(
                primary_candidate["metrics"][
                    "mean_unscaled_net_expectancy_usd_at_full_equity"
                ]
            ),
            position_fractions,
        )
    else:
        meta_candidate = _skipped_meta_candidate(
            str(meta["skip_reason"]), len(primary_event_indices)
        )
        meta_trades = []
        meta_decisions = _skipped_decision_rows(
            research_bars, primary_event_indices, position_fractions
        )
    candidates = [primary_candidate, adx_ablation_candidate, meta_candidate]
    trades = [*primary_trades, *adx_ablation_trades, *meta_trades]
    decisions = [*primary_decisions, *adx_ablation_decisions, *meta_decisions]
    exit_matrix = _run_exit_geometry_matrix(
        research_bars, primary_event_indices, forward_end, position_fractions
    )
    _attach_dsr_diagnostics(candidates, exit_matrix)
    report = _build_report(
        commit_sha,
        provenance,
        bars,
        train_end,
        forward_end,
        signal_counts,
        primary_signal_diagnostics,
        adx_ablation_diagnostics,
        train_gate,
        meta,
        regime_diagnostics,
        candidates,
        trades,
        decisions,
        exit_matrix,
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_trades_csv, trades)
    _write_csv(output_decisions_csv, decisions)
    summary = _render_step_summary(report)
    print(summary, end="")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-trades-csv", type=Path, required=True)
    parser.add_argument("--output-decisions-csv", type=Path, required=True)
    parser.add_argument("--days", type=int, choices=(DAYS,), default=DAYS)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    report = run_v4ai(
        args.input_dir,
        args.output_report,
        args.output_trades_csv,
        args.output_decisions_csv,
        days=args.days,
        commit_sha=args.commit_sha,
    )
    return 0 if len(report["candidates"]) == len(CANDIDATE_NAMES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
