"""V4-AI ETHUSDT 1h maker-fee strategy validation with ADX-filtered primary signals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import kurtosis, norm, skew

import scripts.run_ml_core_v4ac_cost_optimization as v4ac
import scripts.run_ml_core_v4af_meta_final as v4af
from services.meta_labeling import v4ah as advanced
from services.meta_labeling.v4y import (
    V4Y_EMA_TREND_PERIOD,
    V4Y_OOF_FOLD_COUNT,
    generate_v4y_oof_primary_events,
)
from services.ml_core.economic_backtest import simulate_signals

SCHEMA = "alix.ml-core-v4ai-maker-strategy.v1"
SYMBOL = "ETHUSDT"
TIMEFRAME = "1h"
DAYS = 730
INITIAL_EQUITY = 10_000.0
FEE_BPS_PER_SIDE = 10.0
SLIPPAGE_BPS_PER_SIDE = 5.0
ROUND_TRIP_COST_BPS = 30.0
ADX_FEATURE = "%-trend_adx_14"
ADX_PERIOD = 14
ADX_THRESHOLD = 20.0
REGIME_WARMUP_BARS = 200
REGIME_CLUSTER_COUNT = 3
META_TARGET_ACCEPT_FRACTION = 0.70
MINIMUM_TRADES = 30
MINIMUM_NET_PF = 1.5
MINIMUM_WIN_RATE = 0.45
MAXIMUM_DRAWDOWN = 15.0
DSR_MINIMUM_RETURN_OBSERVATIONS = 4
SORTINO_TARGET_RETURN_PER_TRADE = 0.0
BASE_CONTROL = "PRIMARY_BASE_CONTROL_30BPS"
ADX_CONTROL = "PRIMARY_ADX20_30BPS"
META_CANDIDATE = "META_TOP_70_ADX20_30BPS"
CANDIDATE_NAMES = (BASE_CONTROL, ADX_CONTROL, META_CANDIDATE)
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
    "initial_stop_atr": 2.0,
    "take_profit_atr": 4.0,
    "trailing_activation_atr": 1.5,
    "trailing_step_atr": 2.0,
    "horizon_bars": 24.0,
}
EXIT_POLICIES: tuple[dict[str, float | str], ...] = (
    {
        "name": "V4AI_REFERENCE",
        "initial_stop_atr": 2.0,
        "take_profit_atr": 4.0,
        "trailing_activation_atr": 1.0,
        "trailing_step_atr": 1.5,
    },
    {
        "name": "V4AE_HISTORICAL_BEST",
        "initial_stop_atr": 2.0,
        "take_profit_atr": 4.0,
        "trailing_activation_atr": 1.5,
        "trailing_step_atr": 2.0,
    },
    {
        "name": "V4AF_OWNER_GEOMETRY",
        "initial_stop_atr": 2.0,
        "take_profit_atr": 4.0,
        "trailing_activation_atr": 1.5,
        "trailing_step_atr": 1.5,
    },
)


def _validate_request(days: int, commit_sha: str) -> None:
    valid_sha = len(commit_sha) == 40 and all(
        character in "0123456789abcdef" for character in commit_sha
    )
    if days != DAYS or not valid_sha:
        raise ValueError("v4ai_request_identity_invalid")
    if v4af.SYMBOL != SYMBOL or v4af.TIMEFRAME != TIMEFRAME:
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
    )


def _apply_adx_filter(
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
    if event_index + v4af.HORIZON_BARS_VALUE >= train_end:
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
        "oof_fold_count": V4Y_OOF_FOLD_COUNT,
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
    oof_events = generate_v4y_oof_primary_events(
        primary_signal,
        start_index=V4Y_EMA_TREND_PERIOD,
        stop_index=train_end,
        fold_count=V4Y_OOF_FOLD_COUNT,
    )
    labelled, diagnostics = _training_oof_labels(bars, oof_events, train_end)
    return v4af._join_event_features(features, labelled), diagnostics


def _support_shortfall(error: ValueError) -> bool:
    code = str(error)
    prefixes = (
        "v4ah_fit_rows_insufficient",
        "v4ah_fit_rows_insufficient_after_purge",
        "v4ah_temperature_rows_insufficient_after_purge",
        "v4ah_binary_class_support_insufficient",
        "v4ah_shap_feature_support_insufficient",
        "v4ah_temperature_calibration_inputs_invalid",
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
            "skip_reason": "v4ai_insufficient_adx_filtered_primary_signals",
            "separation_diagnostics": {"status": "NOT_AVAILABLE_MODEL_NOT_FIT"},
        }
    try:
        training, training_diagnostics = _oof_training_frame(
            bars, features, primary_signal, train_end
        )
        fit_rows, temperature_rows, threshold_rows, split_diagnostics = (
            advanced.split_fit_temperature_threshold(
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


def _regime_features(features: DataFrame) -> tuple[DataFrame, dict[str, Any]]:
    transformed, diagnostics, _, _ = advanced.build_regime_features(
        features,
        fit_end=REGIME_WARMUP_BARS,
        cluster_count=REGIME_CLUSTER_COUNT,
    )
    return transformed, diagnostics


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
    return f"{SYMBOL}|{TIMEFRAME}|ADX14_GT20|{mode}"


def _profit_factor(metrics: dict[str, Any]) -> float:
    if bool(metrics.get("net_profit_factor_infinite_no_losses")):
        return math.inf
    value = metrics.get("net_profit_factor")
    return -math.inf if value is None else float(value)


def _expectancy(trades: list[dict[str, Any]]) -> float:
    returns = [float(item["net_return_fraction"]) for item in trades]
    return float(np.mean(returns) * INITIAL_EQUITY) if returns else 0.0


def _trade_risk_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray(
        [float(trade["net_return_fraction"]) for trade in trades], dtype=float
    )
    if returns.ndim != 1 or not np.isfinite(returns).all():
        raise ValueError("v4ai_risk_metric_returns_invalid")
    count = int(returns.size)
    metrics: dict[str, Any] = {
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
    if count < 2:
        return metrics
    excess_returns = returns - SORTINO_TARGET_RETURN_PER_TRADE
    downside = np.minimum(excess_returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    if downside_deviation > 0.0:
        metrics["sortino_ratio_per_trade"] = float(
            np.mean(excess_returns) / downside_deviation
        )
        metrics["sortino_status"] = "COMPUTED"
    else:
        metrics["sortino_status"] = "ZERO_DOWNSIDE_DEVIATION"
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
            "return_skewness": float(skew(returns, bias=False)),
            "return_kurtosis_non_excess": float(
                kurtosis(returns, fisher=False, bias=False)
            ),
        }
    )
    return metrics


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
    if len(trial_sharpes) < 2:
        return result
    if (
        int(metrics.get("return_observation_count", 0))
        < DSR_MINIMUM_RETURN_OBSERVATIONS
    ):
        result["dsr_status"] = "INSUFFICIENT_RETURN_OBSERVATIONS"
        return result
    observed_sharpe = metrics.get("sharpe_ratio_per_trade")
    return_skewness = metrics.get("return_skewness")
    return_kurtosis = metrics.get("return_kurtosis_non_excess")
    if observed_sharpe is None or return_skewness is None or return_kurtosis is None:
        result["dsr_status"] = "RETURN_DISTRIBUTION_UNAVAILABLE"
        return result
    trial_values = np.asarray(trial_sharpes, dtype=float)
    if trial_values.ndim != 1 or not np.isfinite(trial_values).all():
        result["dsr_status"] = "TRIAL_SHARPE_DISTRIBUTION_INVALID"
        return result
    trial_count = int(trial_values.size)
    euler_mascheroni = 0.5772156649015329
    expected_max_multiplier = (1.0 - euler_mascheroni) * float(
        norm.ppf(1.0 - 1.0 / trial_count)
    ) + euler_mascheroni * float(norm.ppf(1.0 - 1.0 / (trial_count * math.e)))
    trial_mean = float(np.mean(trial_values))
    trial_std = float(np.std(trial_values, ddof=1))
    expected_max_sharpe = trial_mean + trial_std * expected_max_multiplier
    variance_term = (
        1.0
        - float(return_skewness) * float(observed_sharpe)
        + ((float(return_kurtosis) - 1.0) / 4.0) * float(observed_sharpe) ** 2
    )
    if not math.isfinite(expected_max_sharpe) or not math.isfinite(variance_term):
        result["dsr_status"] = "DSR_VARIANCE_INVALID"
        return result
    if variance_term <= 0.0:
        result["dsr_status"] = "DSR_VARIANCE_INVALID"
        return result
    z_score = (
        (float(observed_sharpe) - expected_max_sharpe)
        * math.sqrt(int(metrics["return_observation_count"]) - 1)
        / math.sqrt(variance_term)
    )
    probability = float(norm.cdf(z_score))
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
                "meta_filter_decision": "ACCEPT" if accepted else "REJECT",
                "executable_signal": "BUY" if accepted else "NO_ENTRY",
                "split_role": "FORWARD_VALIDATION",
            }
        )
    return rows


def _skipped_decision_rows(
    bars: DataFrame, event_indices: list[int]
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
            "meta_filter_decision": "SKIPPED",
            "executable_signal": "NOT_EVALUATED",
            "split_role": "FORWARD_VALIDATION",
        }
        for index in event_indices
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
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = [
        index
        for index in event_indices
        if threshold is None
        or (
            probabilities is not None
            and probabilities.get(index) is not None
            and float(probabilities[index]) >= threshold
        )
    ]
    actions = v4ac._build_forward_actions(bars, accepted, forward_end)
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
    candidate = {
        "candidate_id": _candidate_id(mode),
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "mode": mode,
        "status": "COMPLETED",
        "filter": "ADX14_GT20" if mode != BASE_CONTROL else "NONE",
        "primary_signal_count": len(event_indices),
        "accepted_signal_count": len(accepted),
        "accepted_signal_fraction": (
            len(accepted) / len(event_indices) if event_indices else 0.0
        ),
        "selected_meta_probability_threshold": threshold,
        "metrics": metrics,
        "acceptance_target_met": _candidate_passes(
            metrics, expectancy, primary_metrics, primary_expectancy
        ),
        "sequence_model_used": False,
    }
    enriched = [
        {
            **trade,
            "candidate_id": candidate["candidate_id"],
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
    decisions = _decision_rows(bars, event_indices, probabilities, threshold, mode)
    return candidate, enriched, decisions


def _skipped_meta_candidate(reason: str, event_count: int) -> dict[str, Any]:
    return {
        "candidate_id": _candidate_id(META_CANDIDATE),
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "mode": META_CANDIDATE,
        "status": "SKIPPED_INSUFFICIENT_META_SUPPORT",
        "skip_reason": reason,
        "filter": "ADX14_GT20",
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
) -> dict[str, Any]:
    actions = v4ac._build_forward_actions(bars, event_indices, forward_end)
    execution = _execution_config()
    cells: list[dict[str, Any]] = []
    for policy in EXIT_POLICIES:
        management = _trade_management_config(cast(dict[str, Any], policy))
        replay = simulate_signals(
            bars,
            actions,
            execution,
            trade_management_config=management,
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
    if cell.get("symbol") != SYMBOL or cell.get("timeframe") != TIMEFRAME:
        raise ValueError("v4ai_exit_matrix_cell_invalid")


def _validate_exit_matrix_cell_scope(cell: dict[str, Any]) -> None:
    valid = (
        cell.get("diagnostic_only") is True
        and cell.get("meta_model_used") is False
        and cell.get("final_holdout_used") is False
    )
    if not valid:
        raise ValueError("v4ai_exit_matrix_cell_invalid")


def _validate_exit_matrix_cell(cell: dict[str, Any]) -> None:
    _validate_exit_matrix_cell_cost(cell)
    _validate_exit_matrix_cell_market(cell)
    _validate_exit_matrix_cell_scope(cell)


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


def _validate_report_primary_filter(report: dict[str, Any]) -> None:
    expected = {
        "feature": ADX_FEATURE,
        "period": ADX_PERIOD,
        "threshold": ADX_THRESHOLD,
        "comparison": "STRICT_GT",
    }
    if report.get("primary_filter") != expected:
        raise ValueError("v4ai_report_primary_filter_invalid")


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
    elif separation.get("status") != "NOT_AVAILABLE_MODEL_NOT_FIT":
        raise ValueError("v4ai_separation_status_missing")


def _validate_report(report: dict[str, Any], commit_sha: str) -> None:
    _validate_report_identity(report, commit_sha)
    _validate_report_cost_contract(report)
    _validate_report_risk_contract(report)
    _validate_report_primary_filter(report)
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


def _render_step_summary(report: dict[str, Any]) -> str:
    if report.get("schema_version") != SCHEMA:
        raise ValueError("v4ai_step_summary_schema_invalid")
    lines = [
        "## AliX V4-AI ETHUSDT Maker-Fee Strategy Validation",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "Scope: ETHUSDT Spot, 1h, 730 days, public OHLCV only.",
        "Primary base signal: causal Bollinger(20, 2.0) upper breakout and Close > EMA200.",
        "Primary upgrade: ADX(14) strictly >20; non-finite ADX rows cannot signal.",
        "Execution: closed-candle signal, next-bar-open entry, no pyramiding.",
        "Maker-cost assumption: 10 bps fee + 5 bps slippage per side = 30 bps round trip.",
        "Important: fixed maker-fee proxy only; passive Limit queue/partial-fill mechanics are not simulated.",
        "Risk returns: net closed-trade returns; Sharpe and Sortino are per-trade, not calendar-annualized; Sortino target=0.",
        "DSR: Bailey-López de Prado expected-maximum-Sharpe deflation; DSR Z is unbounded and DSR probability=Phi(Z) is in [0, 1].",
        "DSR trials are limited to each current-run comparison cohort; historical trials are not included and trial independence is unverified.",
        "Z.AI diagnostic reference (not a promotion gate): Net PF >1.1 and DSR Z >1.0; the existing Net PF >1.5 acceptance gate remains unchanged.",
        "Fixed candidate geometry: SL=2.0 ATR, TP=4.0 ATR, activation=1.5 ATR, step=2.0 ATR, horizon=24 bars.",
        "Training: outer 70%; purged OOF Fit/Temperature/Threshold slices=70/15/15.",
        "Evaluation: next 15% forward validation; final 15% holdout untouched.",
        "Meta: LightGBM + TreeSHAP top-30 Fit-only selection; KMeans(3) regimes as features.",
        "Meta cutoff: accept the upper 70% of probabilities using the 30th percentile order statistic on the third purged OOF slice; ties are included.",
        "",
        "| Candidate | Signals | Accepted | Trades | Win Rate % | Net PF | Max DD % | DSR Z | DSR P | DSR N | Sortino/trade | Z.AI PF>1.1 | Z.AI DSR Z>1 | Net PnL USD | Net Expectancy USD/trade | Acceptance | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|",
    ]
    for item in report["candidates"]:
        metrics = item["metrics"]
        win_rate = metrics.get("win_rate")
        drawdown = metrics.get("max_drawdown_percent")
        expectancy = metrics.get("mean_unscaled_net_expectancy_usd_at_full_equity")
        lines.append(
            f"| {item['mode']} | {item.get('primary_signal_count', 0)} "
            f"| {item.get('accepted_signal_count', 0)} "
            f"| {metrics.get('trade_count', 0)} "
            f"| {'N/A' if win_rate is None else f'{float(win_rate) * 100.0:.2f}'} "
            f"| {_format_pf(metrics)} "
            f"| {'N/A' if drawdown is None else f'{float(drawdown):.2f}'} "
            f"| {_format_optional_ratio(metrics.get('dsr_z_score'))} "
            f"| {_format_optional_ratio(metrics.get('dsr_probability'))} "
            f"| {metrics.get('dsr_trial_count', 'N/A')} "
            f"| {_format_optional_ratio(metrics.get('sortino_ratio_per_trade'))} "
            f"| {_format_diagnostic_flag(metrics.get('zai_net_pf_gt_1_1'))} "
            f"| {_format_diagnostic_flag(metrics.get('zai_dsr_z_gt_1_0'))} "
            f"| {float(metrics.get('net_pnl_total_usd') or 0.0):.2f} "
            f"| {'N/A' if expectancy is None else f'{float(expectancy):.4f}'} "
            f"| {'PASS' if item.get('acceptance_target_met') else 'NO'} "
            f"| {item['status']} |"
        )
    meta = report.get("meta_model", {})
    separation = meta.get("separation_diagnostics", {})
    fit = meta.get("fit_diagnostics", {})
    threshold = meta.get("threshold_diagnostics", {})
    lines.extend(
        [
            "",
            f"Base Primary BUYs: {report['signal_counts']['base_primary_buy_count']}; ADX-filtered Primary BUYs: {report['signal_counts']['adx_filtered_primary_buy_count']}.",
            f"OOF labels: {meta.get('training_diagnostics', {}).get('matured_costed_oof_label_count', 0)}.",
            (
                "Purged OOF slices after label-end purge: "
                f"fit={meta.get('split_diagnostics', {}).get('fit_rows', 'N/A')}; "
                f"temperature={meta.get('split_diagnostics', {}).get('temperature_rows', 'N/A')}; "
                f"threshold={meta.get('split_diagnostics', {}).get('threshold_rows', 'N/A')}."
            ),
            f"Meta fit: {'ENABLED' if meta.get('enabled') else 'SKIPPED'}; reason={meta.get('skip_reason') or 'N/A'}.",
            (
                f"Regime: KMeans clusters={report.get('regime_diagnostics', {}).get('cluster_count', 'N/A')}; "
                f"fit rows={report.get('regime_diagnostics', {}).get('fit_rows', 0)}."
            ),
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
            "",
            "### ADX-filtered Primary exit-geometry matrix (30 bps only)",
            "| Policy | SL ATR | TP ATR | Activation ATR | Step ATR | Cost bps RT | Trades | Win % | Net PF | Max DD % | DSR Z | DSR P | DSR N | Sortino/trade | Z.AI PF>1.1 | Z.AI DSR Z>1 | Net PnL USD | Expectancy USD/trade |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
        ]
    )
    for cell in report["exit_geometry_matrix"]["cells"]:
        win_rate = cell["win_rate"]
        win_rate_text = "N/A" if win_rate is None else f"{float(win_rate) * 100.0:.2f}"
        lines.append(
            f"| {cell['policy']} | {float(cell['initial_stop_atr']):.1f} "
            f"| {float(cell['take_profit_atr']):.1f} "
            f"| {float(cell['trailing_activation_atr']):.1f} "
            f"| {float(cell['trailing_step_atr']):.1f} "
            f"| {float(cell['round_trip_cost_bps']):.0f} "
            f"| {cell['trade_count']} "
            f"| {win_rate_text} "
            f"| {_format_pf(cell)} "
            f"| {float(cell['max_drawdown_percent']):.2f} "
            f"| {_format_optional_ratio(cell.get('dsr_z_score'))} "
            f"| {_format_optional_ratio(cell.get('dsr_probability'))} "
            f"| {cell.get('dsr_trial_count', 'N/A')} "
            f"| {_format_optional_ratio(cell.get('sortino_ratio_per_trade'))} "
            f"| {_format_diagnostic_flag(cell.get('zai_net_pf_gt_1_1'))} "
            f"| {_format_diagnostic_flag(cell.get('zai_dsr_z_gt_1_0'))} "
            f"| {float(cell['net_pnl_total_usd']):.2f} "
            f"| {float(cell['mean_unscaled_net_expectancy_usd_at_full_equity']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Acceptance: Net PF >1.5, Max DD <15%, Win Rate >45%, Trades >=30; Meta must also outperform ADX Primary.",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0, sequence_model_used=false.",
            "Research only. No private exchange API, credentials, live/paper orders, or production execution.",
            "Promotion: BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW.",
        ]
    )
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
    adx_diagnostics: dict[str, Any],
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
        "primary_signal": "CAUSAL_BOLLINGER_20_UPPER_BREAKOUT_CLOSE_ABOVE_EMA200",
        "primary_filter": {
            "feature": ADX_FEATURE,
            "period": ADX_PERIOD,
            "threshold": ADX_THRESHOLD,
            "comparison": "STRICT_GT",
        },
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
        "adx_filter_diagnostics": adx_diagnostics,
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
        base_primary_signal,
    ) = v4af._load_research_context(input_dir, days, commit_sha)
    adx_signal, adx_diagnostics = _apply_adx_filter(base_primary_signal, base_features)
    base_event_indices = v4af._eligible_primary_event_indices(
        base_primary_signal, train_end, forward_end
    )
    signal_counts = {
        "base_primary_buy_count": int(base_primary_signal.sum()),
        "adx_filtered_primary_buy_count": int(adx_signal.sum()),
        "base_forward_event_count": len(base_event_indices),
        "adx_filtered_forward_event_count": len(
            v4af._eligible_primary_event_indices(adx_signal, train_end, forward_end)
        ),
    }
    _, train_gate, adx_event_indices = v4af._primary_signal_evaluation(
        adx_signal, train_end, forward_end
    )
    base_candidate, base_trades, base_decisions = _run_candidate(
        research_bars,
        base_event_indices,
        forward_end,
        BASE_CONTROL,
        None,
        None,
    )
    adx_candidate, adx_trades, adx_decisions = _run_candidate(
        research_bars,
        adx_event_indices,
        forward_end,
        ADX_CONTROL,
        None,
        None,
    )
    try:
        features, regime_diagnostics = _regime_features(base_features)
        meta = _fit_meta_model(
            research_bars, features, adx_signal, train_end, train_gate
        )
    except ValueError as error:
        if not _support_shortfall(error):
            raise
        meta = _empty_meta(str(error))
        regime_diagnostics = {"status": "SKIPPED", "reason": str(error)}
    if meta.get("skip_reason") is None:
        probabilities = _forward_probability_map(
            features,
            adx_event_indices,
            cast(list[str], meta["selected_features"]),
            meta["model"],
            float(meta["temperature"]),
        )
        meta_candidate, meta_trades, meta_decisions = _run_candidate(
            research_bars,
            adx_event_indices,
            forward_end,
            META_CANDIDATE,
            float(meta["threshold"]),
            probabilities,
            adx_candidate["metrics"],
            float(
                adx_candidate["metrics"][
                    "mean_unscaled_net_expectancy_usd_at_full_equity"
                ]
            ),
        )
    else:
        meta_candidate = _skipped_meta_candidate(
            str(meta["skip_reason"]), len(adx_event_indices)
        )
        probabilities = {}
        meta_trades = []
        meta_decisions = _skipped_decision_rows(research_bars, adx_event_indices)
    candidates = [base_candidate, adx_candidate, meta_candidate]
    trades = [*base_trades, *adx_trades, *meta_trades]
    decisions = [*base_decisions, *adx_decisions, *meta_decisions]
    exit_matrix = _run_exit_geometry_matrix(
        research_bars, adx_event_indices, forward_end
    )
    _attach_dsr_diagnostics(candidates, exit_matrix)
    report = _build_report(
        commit_sha,
        provenance,
        bars,
        train_end,
        forward_end,
        signal_counts,
        adx_diagnostics,
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
