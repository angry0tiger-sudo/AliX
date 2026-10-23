"""Deterministic, offline-only walk-forward evaluation for the Phase 4 baseline."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from pandas import DataFrame

from services.data_quality.gate import validate_candles
from services.freqai_baseline.features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    OHLCV_COLUMNS,
    TARGET_COLUMN,
    TARGET_SCHEMA_VERSION,
    add_market_features,
    forward_return_target,
)

BASELINE_MODEL_PARAMETERS: Final[dict[str, object]] = {
    "objective": "regression",
    "n_estimators": 80,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}
REPORT_SCHEMA_VERSION: Final[str] = "alix.freqai.walk-forward-report.v1"


def _validate_integer_parameter(name: str, value: object, minimum: int) -> None:
    """Reject bools, non-integers, and values below a named minimum."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _validate_cost_parameter(name: str, value: float) -> None:
    """Reject negative and non-finite transaction-cost assumptions."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Predeclared split and transaction-cost assumptions for research runs."""

    horizon: int = 12
    n_splits: int = 3
    min_train_size: int = 300
    fee_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0

    def __post_init__(self) -> None:
        _validate_integer_parameter("horizon", self.horizon, 1)
        _validate_integer_parameter("n_splits", self.n_splits, 2)
        _validate_integer_parameter("min_train_size", self.min_train_size, 30)
        _validate_cost_parameter("fee_bps_per_side", self.fee_bps_per_side)
        _validate_cost_parameter("slippage_bps_per_side", self.slippage_bps_per_side)


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """Position-based fold boundaries; train_stop/test_stop are exclusive."""

    fold: int
    train_start: int
    train_stop: int
    test_start: int
    test_stop: int


EstimatorFactory = Callable[..., Any]


def validate_ohlcv(
    dataframe: DataFrame, timeframe_seconds: int
) -> tuple[DataFrame, str]:
    """Apply the Phase 3 gate without repairing, sorting, or imputing candles."""
    missing = sorted(set(OHLCV_COLUMNS) - set(dataframe.columns))
    if missing:
        raise ValueError(f"OHLCV input is missing required columns: {missing}")
    if dataframe.columns.duplicated().any():
        raise ValueError("OHLCV input contains duplicate column names")
    raw_rows = dataframe.loc[:, list(OHLCV_COLUMNS)].to_dict(orient="records")
    rows: list[dict[str, object]] = [
        {str(key): value for key, value in row.items()} for row in raw_rows
    ]
    result = validate_candles(
        rows,
        expected_interval_seconds=timeframe_seconds,
        gap_policy="reject",
    )
    if not result.passed:
        codes = sorted({code.value for code in result.reason_codes})
        raise ValueError(f"Data Quality Gate rejected the dataset: {codes}")
    return dataframe.copy(), result.fingerprint or hashlib.sha256(b"").hexdigest()


def prepare_model_dataset(dataframe: DataFrame, horizon: int) -> DataFrame:
    """Create finite feature/label rows while preserving their chronological order."""
    engineered = add_market_features(dataframe)
    engineered[TARGET_COLUMN] = forward_return_target(engineered, horizon)
    required = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    numeric = engineered[required].replace([np.inf, -np.inf], np.nan)
    valid = numeric.notna().all(axis=1)
    selected_columns = list(
        dict.fromkeys(["date", "close", "%-volatility_12", *required])
    )
    dataset = engineered.loc[valid, selected_columns].copy()
    return dataset.reset_index(drop=True)


def make_walk_forward_folds(
    sample_count: int, config: WalkForwardConfig
) -> list[WalkForwardFold]:
    """Build expanding, strictly temporal folds with a label-horizon purge."""
    first_test_start = config.min_train_size + config.horizon
    available = sample_count - first_test_start
    if available < config.n_splits:
        raise ValueError(
            "insufficient valid rows for the configured training size, horizon, and folds"
        )
    folds: list[WalkForwardFold] = []
    for index in range(config.n_splits):
        test_start = first_test_start + (available * index) // config.n_splits
        test_stop = first_test_start + (available * (index + 1)) // config.n_splits
        train_stop = test_start - config.horizon
        if train_stop < config.min_train_size or test_stop <= test_start:
            raise ValueError("walk-forward split produced an invalid fold")
        folds.append(
            WalkForwardFold(
                fold=index + 1,
                train_start=0,
                train_stop=train_stop,
                test_start=test_start,
                test_stop=test_stop,
            )
        )
    return folds


def _new_estimator(
    factory: EstimatorFactory | None, parameters: dict[str, object]
) -> Any:
    if factory is not None:
        return factory(**parameters)
    from lightgbm import LGBMRegressor  # type: ignore[import-not-found]

    return LGBMRegressor(**parameters)


def _forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "directional_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual))),
    }


def _max_drawdown(net_returns: list[float]) -> float:
    equity = np.cumprod(1.0 + np.asarray(net_returns, dtype=float))
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    return float(np.min(equity / peaks - 1.0))


def _simulate_non_overlapping_trades(
    test: DataFrame,
    predictions: np.ndarray,
    config: WalkForwardConfig,
) -> dict[str, Any]:
    """Evaluate a fixed-horizon long-only research rule; never create orders."""
    fee = 2.0 * config.fee_bps_per_side / 10_000.0
    slippage = 2.0 * config.slippage_bps_per_side / 10_000.0
    round_trip_cost = fee + slippage
    net_returns: list[float] = []
    gross_returns: list[float] = []
    trade_count = 0
    last_safe_offset = len(test) - config.horizon
    for offset in range(0, max(0, last_safe_offset + 1), config.horizon):
        forecast = float(predictions[offset])
        if forecast <= round_trip_cost:
            continue
        realized = float(test[TARGET_COLUMN].iloc[offset])
        gross_returns.append(realized)
        net_returns.append(realized - round_trip_cost)
        trade_count += 1
    net_total = float(np.prod(1.0 + np.asarray(net_returns)) - 1.0)
    gross_total = float(np.prod(1.0 + np.asarray(gross_returns)) - 1.0)
    if not net_returns:
        net_total = 0.0
        gross_total = 0.0
    buy_hold_gross = float(test["close"].iloc[-1] / test["close"].iloc[0] - 1.0)
    buy_hold_net = buy_hold_gross - round_trip_cost if len(test) > 1 else 0.0
    return {
        "gross_return": gross_total,
        "net_return": net_total,
        "max_drawdown": _max_drawdown(net_returns),
        "turnover_units": float(2 * trade_count),
        "fees_bps_cumulative": float(2 * trade_count * config.fee_bps_per_side),
        "slippage_bps_cumulative": float(
            2 * trade_count * config.slippage_bps_per_side
        ),
        "trade_count": trade_count,
        "winning_trade_count": sum(value > 0 for value in net_returns),
        "losing_trade_count": sum(value <= 0 for value in net_returns),
        "win_rate": (sum(value > 0 for value in net_returns) / trade_count)
        if trade_count
        else None,
        "expectancy_net_per_trade": (sum(net_returns) / trade_count)
        if trade_count
        else None,
        "net_trade_returns": net_returns,
        "buy_and_hold_gross_return": buy_hold_gross,
        "buy_and_hold_net_return": buy_hold_net,
        "no_trade_net_return": 0.0,
    }


def _regime_metrics(
    train: DataFrame, test: DataFrame, actual: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    threshold = float(train["%-volatility_12"].median())
    volatility = test["%-volatility_12"].to_numpy(dtype=float)
    result: dict[str, Any] = {"training_volatility_median": threshold, "groups": {}}
    for name, mask in (
        ("low_volatility", volatility <= threshold),
        ("high_volatility", volatility > threshold),
    ):
        if not bool(mask.any()):
            result["groups"][name] = {"sample_count": 0}
            continue
        metrics = _forecast_metrics(actual[mask], predicted[mask])
        metrics["sample_count"] = int(mask.sum())
        result["groups"][name] = metrics
    return result


def _roundtrip_model(
    model: Any, features: DataFrame, expected: np.ndarray
) -> dict[str, Any]:
    """Reload from a model-native serialization contract without unsafe pickle."""
    booster = getattr(model, "booster_", None)
    if booster is not None and callable(getattr(booster, "model_to_string", None)):
        from lightgbm import Booster

        payload = booster.model_to_string()
        restored = Booster(model_str=payload)
        serializer = "lightgbm-native-model-string"
    else:
        serializer_method = getattr(model, "model_to_string", None)
        restore_method = getattr(type(model), "model_from_string", None)
        if not callable(serializer_method) or not callable(restore_method):
            raise TypeError(
                "estimator must expose a native model serialization and restore contract"
            )
        payload = serializer_method()
        restored = restore_method(payload)
        serializer = "estimator-native-model-string"

    restored_predictions = np.asarray(restored.predict(features), dtype=float)
    max_difference = (
        float(np.max(np.abs(restored_predictions - expected))) if expected.size else 0.0
    )
    if not np.allclose(restored_predictions, expected, rtol=1e-12, atol=1e-12):
        raise RuntimeError("model predictions changed after artifact reload")
    return {
        "serializer": serializer,
        "artifact_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "reload_roundtrip": "passed",
        "roundtrip_prediction_max_abs_diff": max_difference,
    }


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _timestamp(value: object) -> str:
    return pd.Timestamp(cast(Any, value)).isoformat()


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    """Research evidence emitted by one temporal fold; no execution side effects."""

    report: dict[str, Any]
    actual: np.ndarray
    predicted: np.ndarray
    trade_returns: list[float]
    economic: dict[str, Any]
    training_seconds: float
    prediction_latency_ms_per_row: float
    model: Any
    test_features: DataFrame


def _evaluate_fold(
    dataset: DataFrame,
    fold: WalkForwardFold,
    parameters: dict[str, object],
    config: WalkForwardConfig,
    estimator_factory: EstimatorFactory | None,
) -> FoldEvaluation:
    train = dataset.iloc[fold.train_start : fold.train_stop]
    test = dataset.iloc[fold.test_start : fold.test_stop]
    train_features = train.loc[:, list(FEATURE_COLUMNS)]
    test_features = test.loc[:, list(FEATURE_COLUMNS)]
    actual = test[TARGET_COLUMN].to_numpy(dtype=float)
    model = _new_estimator(estimator_factory, parameters)
    started = perf_counter()
    model.fit(train_features, train[TARGET_COLUMN].to_numpy(dtype=float))
    training_seconds = perf_counter() - started
    started = perf_counter()
    predicted = np.asarray(model.predict(test_features), dtype=float)
    prediction_seconds = perf_counter() - started
    if len(predicted) != len(actual) or not np.isfinite(predicted).all():
        raise ValueError("model returned malformed or non-finite predictions")
    economic = _simulate_non_overlapping_trades(test, predicted, config)
    fold_report = {
        "fold": fold.fold,
        "train_rows": len(train),
        "test_rows": len(test),
        "purged_label_rows": config.horizon,
        "train_start": _timestamp(train["date"].iloc[0]),
        "train_end": _timestamp(train["date"].iloc[-1]),
        "test_start": _timestamp(test["date"].iloc[0]),
        "test_end": _timestamp(test["date"].iloc[-1]),
        "forecast_oos": _forecast_metrics(actual, predicted),
        "zero_forecast_baseline_oos": _forecast_metrics(actual, np.zeros_like(actual)),
        "costed_fixed_horizon_research_rule": economic,
        "volatility_regime_oos": _regime_metrics(train, test, actual, predicted),
    }
    return FoldEvaluation(
        report=fold_report,
        actual=actual,
        predicted=predicted,
        trade_returns=_collect_trade_returns(test, predicted, config),
        economic=economic,
        training_seconds=training_seconds,
        prediction_latency_ms_per_row=prediction_seconds
        * 1000.0
        / max(len(test_features), 1),
        model=model,
        test_features=test_features,
    )


def _economic_gate_status(
    sample_count: int,
    win_rate: float | None,
    max_drawdown: float,
    expectancy: float | None,
) -> str:
    if sample_count < 100:
        return "INSUFFICIENT_SAMPLE"
    if win_rate is None or win_rate < 0.60:
        return "NO_GO_THRESHOLDS_NOT_MET"
    if abs(max_drawdown) > 0.15:
        return "NO_GO_THRESHOLDS_NOT_MET"
    if expectancy is None or expectancy <= 0.0:
        return "NO_GO_THRESHOLDS_NOT_MET"
    return "THRESHOLDS_MET_RESEARCH_ONLY"


def _economic_summary(
    dataset: DataFrame,
    fold_results: list[FoldEvaluation],
    trade_returns: list[float],
) -> dict[str, Any]:
    trade_count = len(trade_returns)
    winning = sum(result > 0.0 for result in trade_returns)
    win_rate = winning / trade_count if trade_count else None
    expectancy = float(np.mean(trade_returns)) if trade_count else None
    max_drawdown = _max_drawdown(trade_returns)
    return {
        "net_return": float(np.prod(1.0 + np.asarray(trade_returns)) - 1.0)
        if trade_count
        else 0.0,
        "max_drawdown": max_drawdown,
        "trade_count": sum(
            int(result.economic["trade_count"]) for result in fold_results
        ),
        "winning_trade_count": winning,
        "win_rate": win_rate,
        "win_rate_sample_size": trade_count,
        "win_rate_minimum_sample_met": trade_count >= 100,
        "win_rate_time_period": {
            "start": _timestamp(dataset["date"].iloc[0]),
            "end": _timestamp(dataset["date"].iloc[-1]),
        },
        "expectancy_net_per_trade": expectancy,
        "economic_threshold_evaluation": {
            "minimum_trade_count": 100,
            "win_rate_minimum": 0.60,
            "maximum_drawdown_absolute": 0.15,
            "expectancy_must_be_positive": True,
            "win_rate_pass": win_rate is not None and win_rate >= 0.60,
            "drawdown_pass": abs(max_drawdown) <= 0.15,
            "expectancy_pass": expectancy is not None and expectancy > 0.0,
            "status": _economic_gate_status(
                trade_count, win_rate, max_drawdown, expectancy
            ),
            "authorization_effect": "NONE; Champion remains null and promotion/trading remain NO-GO",
        },
    }


def _compose_walk_forward_report(
    *,
    validated: DataFrame,
    gate_fingerprint: str,
    dataset: DataFrame,
    dataset_kind: str,
    dataset_sha256: str,
    pair: str,
    timeframe: str,
    config: WalkForwardConfig,
    folds: list[WalkForwardFold],
    parameters: dict[str, object],
    experiment_config_sha256: str,
    commit_sha: str,
    evaluations: list[FoldEvaluation],
    oos_actual: np.ndarray,
    oos_predicted: np.ndarray,
    reload_info: dict[str, Any],
    economic: dict[str, Any],
    turnover: float,
    fees: float,
    slippage: float,
    buy_hold_total: float,
) -> dict[str, Any]:
    fold_reports = [result.report for result in evaluations]
    latencies = [result.prediction_latency_ms_per_row for result in evaluations]
    training_times = [result.training_seconds for result in evaluations]
    out_of_sample = {
        "sample_count": len(oos_actual),
        "forecast": _forecast_metrics(oos_actual, oos_predicted),
        "zero_forecast_baseline": _forecast_metrics(
            oos_actual, np.zeros_like(oos_actual)
        ),
        "costed_fixed_horizon_research_rule": {
            **economic,
            "turnover_units": turnover,
            "fees_bps_cumulative": fees,
            "slippage_bps_cumulative": slippage,
            "buy_and_hold_net_return": buy_hold_total,
            "no_trade_net_return": 0.0,
            "execution_boundary": "offline research only; not connected to AliXStrategy or Freqtrade order lifecycle",
        },
        "positive_net_return_fold_count": sum(
            float(item["costed_fixed_horizon_research_rule"]["net_return"]) > 0.0
            for item in fold_reports
        ),
        "fold_net_returns": [
            item["costed_fixed_horizon_research_rule"]["net_return"]
            for item in fold_reports
        ],
    }
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_status": "SYNTHETIC_MECHANICS_ONLY"
        if dataset_kind == "synthetic"
        else "RESEARCH_ONLY_ECONOMIC_GATES_PENDING",
        "commit_sha": commit_sha,
        "experiment_config_sha256": experiment_config_sha256,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "freqtrade": _installed_version("freqtrade"),
            "lightgbm": _installed_version("lightgbm"),
            "scikit_learn": _installed_version("scikit-learn"),
        },
        "pair": pair,
        "timeframe": timeframe,
        "dataset": {
            "kind": dataset_kind,
            "sha256": dataset_sha256,
            "gate_fingerprint": gate_fingerprint,
            "input_rows": len(validated),
            "valid_feature_target_rows": len(dataset),
            "first_candle": _timestamp(validated["date"].iloc[0]),
            "last_candle": _timestamp(validated["date"].iloc[-1]),
        },
        "feature_contract": {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "columns": list(FEATURE_COLUMNS),
            "availability_rule": "current closed candle and past rows only; no future/centered windows",
        },
        "target_contract": {
            "schema_version": TARGET_SCHEMA_VERSION,
            "column": TARGET_COLUMN,
            "horizon_candles": config.horizon,
            "definition": "close[t+horizon] / close[t] - 1; target only, never a feature",
        },
        "split_policy": {
            **asdict(config),
            "method": "expanding walk-forward, chronological, no shuffle",
            "purge_embargo_rows": config.horizon,
            "fold_count": len(folds),
        },
        "model": {
            "class": "AliXBaselineRegressor",
            "algorithm": "LightGBM LGBMRegressor",
            "parameters": parameters,
            **reload_info,
        },
        "out_of_sample": out_of_sample,
        "folds": fold_reports,
        "compute": {
            "training_duration_seconds_total": float(sum(training_times)),
            "training_duration_seconds_by_fold": training_times,
            "prediction_latency_ms_per_row_by_fold": latencies,
            "prediction_latency_ms_per_row_p50": float(np.percentile(latencies, 50)),
            "prediction_latency_ms_per_row_p95": float(np.percentile(latencies, 95)),
        },
        "strategy_scope": {
            "evaluated_rule": "fixed-horizon long-only walk-forward research rule",
            "freqtrade_strategy_backtest": "NOT_RUN: the committed AliX strategy scaffolds intentionally emit no trade signals",
            "risk_engine_end_to_end": "NOT_EVALUATED: public candles are not reconciled account snapshots",
            "orders_or_execution_api_used": False,
        },
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "future_risk_inputs": {
            "regime_probabilities": None,
            "calibrated_confidence": None,
            "calibrated_uncertainty": None,
            "note": "Not produced or accepted for sizing in Phase 4 baseline.",
        },
        "economic_gates": {
            "status": "NOT_EVALUATED",
            "reason": "Owner-approved pre-registered Phase 0 economic acceptance thresholds and authorized historical dataset evidence are not present in this run.",
            "champion_selection": "PROHIBITED",
        },
        "champion": None,
        "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
    }


def run_walk_forward(
    dataframe: DataFrame,
    *,
    pair: str,
    timeframe: str,
    timeframe_seconds: int,
    config: WalkForwardConfig,
    dataset_kind: str,
    dataset_sha256: str,
    commit_sha: str,
    estimator_factory: EstimatorFactory | None = None,
    model_parameters: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Run offline walk-forward forecasting and costed research metrics only."""
    validated, gate_fingerprint = validate_ohlcv(dataframe, timeframe_seconds)
    dataset = prepare_model_dataset(validated, config.horizon)
    folds = make_walk_forward_folds(len(dataset), config)
    parameters = dict(BASELINE_MODEL_PARAMETERS)
    parameters.update(model_parameters or {})
    experiment_config = {"walk_forward": asdict(config), "model_parameters": parameters}
    experiment_sha = hashlib.sha256(
        json.dumps(experiment_config, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    evaluations = [
        _evaluate_fold(dataset, fold, parameters, config, estimator_factory)
        for fold in folds
    ]
    if not evaluations:
        raise RuntimeError("walk-forward evaluation produced no fitted models")
    trade_returns = [value for result in evaluations for value in result.trade_returns]
    actual = np.concatenate([result.actual for result in evaluations])
    predicted = np.concatenate([result.predicted for result in evaluations])
    final = evaluations[-1]
    reload_info = _roundtrip_model(final.model, final.test_features, final.predicted)
    economic = _economic_summary(dataset, evaluations, trade_returns)
    buy_hold_values = [
        float(result.economic["buy_and_hold_net_return"]) for result in evaluations
    ]
    buy_hold_total = float(np.prod(1.0 + np.asarray(buy_hold_values)) - 1.0)
    return _compose_walk_forward_report(
        validated=validated,
        gate_fingerprint=gate_fingerprint,
        dataset=dataset,
        dataset_kind=dataset_kind,
        dataset_sha256=dataset_sha256,
        pair=pair,
        timeframe=timeframe,
        config=config,
        folds=folds,
        parameters=parameters,
        experiment_config_sha256=experiment_sha,
        commit_sha=commit_sha,
        evaluations=evaluations,
        oos_actual=actual,
        oos_predicted=predicted,
        reload_info=reload_info,
        economic=economic,
        turnover=sum(
            float(result.economic["turnover_units"]) for result in evaluations
        ),
        fees=sum(
            float(result.economic["fees_bps_cumulative"]) for result in evaluations
        ),
        slippage=sum(
            float(result.economic["slippage_bps_cumulative"]) for result in evaluations
        ),
        buy_hold_total=buy_hold_total,
    )


def _collect_trade_returns(
    test: DataFrame, predictions: np.ndarray, config: WalkForwardConfig
) -> list[float]:
    cost = 2.0 * (config.fee_bps_per_side + config.slippage_bps_per_side) / 10_000.0
    returns: list[float] = []
    last_safe_offset = len(test) - config.horizon
    for offset in range(0, max(0, last_safe_offset + 1), config.horizon):
        if float(predictions[offset]) > cost:
            returns.append(float(test[TARGET_COLUMN].iloc[offset]) - cost)
    return returns
