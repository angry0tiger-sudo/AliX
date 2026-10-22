"""Deterministic, offline-only walk-forward evaluation for the Phase 4 baseline."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Final

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


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Predeclared split and transaction-cost assumptions for research runs."""

    horizon: int = 12
    n_splits: int = 3
    min_train_size: int = 300
    fee_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if isinstance(self.n_splits, bool) or self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if isinstance(self.min_train_size, bool) or self.min_train_size < 30:
            raise ValueError("min_train_size must be at least 30")
        if self.fee_bps_per_side < 0 or self.slippage_bps_per_side < 0:
            raise ValueError("fee and slippage assumptions cannot be negative")


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
    rows = dataframe.loc[:, list(OHLCV_COLUMNS)].to_dict(orient="records")
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
    dataset = engineered.loc[
        valid, ["date", "close", "%-volatility_12"] + required
    ].copy()
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
        "directional_accuracy": float(
            np.mean(np.sign(predicted) == np.sign(actual))
        ),
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
        "slippage_bps_cumulative": float(2 * trade_count * config.slippage_bps_per_side),
        "trade_count": trade_count,
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
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(payload)
    restored_predictions = np.asarray(restored.predict(features), dtype=float)
    max_difference = (
        float(np.max(np.abs(restored_predictions - expected)))
        if expected.size
        else 0.0
    )
    if not np.allclose(restored_predictions, expected, rtol=1e-12, atol=1e-12):
        raise RuntimeError("model predictions changed after artifact reload")
    return {
        "serializer": "python-pickle",
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "reload_roundtrip": "passed",
        "roundtrip_prediction_max_abs_diff": max_difference,
    }


def _timestamp(value: object) -> str:
    return pd.Timestamp(value).isoformat()


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
    if model_parameters:
        parameters.update(model_parameters)
    all_actual: list[float] = []
    all_predicted: list[float] = []
    all_trade_returns: list[float] = []
    fold_reports: list[dict[str, Any]] = []
    final_model: Any = None
    final_test_features = DataFrame()
    final_predictions = np.asarray([], dtype=float)
    aggregate_trade_count = 0
    aggregate_turnover = 0.0
    aggregate_fees = 0.0
    aggregate_slippage = 0.0
    aggregate_buy_hold_returns: list[float] = []

    for fold in folds:
        train = dataset.iloc[fold.train_start : fold.train_stop]
        test = dataset.iloc[fold.test_start : fold.test_stop]
        train_features = train.loc[:, list(FEATURE_COLUMNS)]
        test_features = test.loc[:, list(FEATURE_COLUMNS)]
        actual = test[TARGET_COLUMN].to_numpy(dtype=float)
        model = _new_estimator(estimator_factory, parameters)
        model.fit(train_features, train[TARGET_COLUMN].to_numpy(dtype=float))
        predicted = np.asarray(model.predict(test_features), dtype=float)
        if len(predicted) != len(actual) or not np.isfinite(predicted).all():
            raise ValueError("model returned malformed or non-finite predictions")

        forecast_metrics = _forecast_metrics(actual, predicted)
        zero_metrics = _forecast_metrics(actual, np.zeros_like(actual))
        economic = _simulate_non_overlapping_trades(test, predicted, config)
        regime = _regime_metrics(train, test, actual, predicted)
        fold_report = {
            "fold": fold.fold,
            "train_rows": len(train),
            "test_rows": len(test),
            "purged_label_rows": config.horizon,
            "train_start": _timestamp(train["date"].iloc[0]),
            "train_end": _timestamp(train["date"].iloc[-1]),
            "test_start": _timestamp(test["date"].iloc[0]),
            "test_end": _timestamp(test["date"].iloc[-1]),
            "forecast_oos": forecast_metrics,
            "zero_forecast_baseline_oos": zero_metrics,
            "costed_fixed_horizon_research_rule": economic,
            "volatility_regime_oos": regime,
        }
        fold_reports.append(fold_report)
        all_actual.extend(actual.tolist())
        all_predicted.extend(predicted.tolist())
        all_trade_returns.extend(
            _collect_trade_returns(test, predicted, config)
        )
        aggregate_trade_count += int(economic["trade_count"])
        aggregate_turnover += float(economic["turnover_units"])
        aggregate_fees += float(economic["fees_bps_cumulative"])
        aggregate_slippage += float(economic["slippage_bps_cumulative"])
        aggregate_buy_hold_returns.append(float(economic["buy_and_hold_net_return"]))
        final_model = model
        final_test_features = test_features
        final_predictions = predicted

    if final_model is None:
        raise RuntimeError("walk-forward evaluation produced no fitted models")
    reload_info = _roundtrip_model(final_model, final_test_features, final_predictions)
    oos_actual = np.asarray(all_actual, dtype=float)
    oos_predicted = np.asarray(all_predicted, dtype=float)
    model_net_total = float(np.prod(1.0 + np.asarray(all_trade_returns)) - 1.0)
    buy_hold_total = float(np.prod(1.0 + np.asarray(aggregate_buy_hold_returns)) - 1.0)
    if not all_trade_returns:
        model_net_total = 0.0

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_status": (
            "SYNTHETIC_MECHANICS_ONLY"
            if dataset_kind == "synthetic"
            else "RESEARCH_ONLY_ECONOMIC_GATES_PENDING"
        ),
        "commit_sha": commit_sha,
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
        "out_of_sample": {
            "sample_count": len(oos_actual),
            "forecast": _forecast_metrics(oos_actual, oos_predicted),
            "zero_forecast_baseline": _forecast_metrics(
                oos_actual, np.zeros_like(oos_actual)
            ),
            "costed_fixed_horizon_research_rule": {
                "net_return": model_net_total,
                "max_drawdown": _max_drawdown(all_trade_returns),
                "trade_count": aggregate_trade_count,
                "turnover_units": aggregate_turnover,
                "fees_bps_cumulative": aggregate_fees,
                "slippage_bps_cumulative": aggregate_slippage,
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
        },
        "folds": fold_reports,
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


def _collect_trade_returns(
    test: DataFrame, predictions: np.ndarray, config: WalkForwardConfig
) -> list[float]:
    cost = 2.0 * (
        config.fee_bps_per_side + config.slippage_bps_per_side
    ) / 10_000.0
    returns: list[float] = []
    last_safe_offset = len(test) - config.horizon
    for offset in range(0, max(0, last_safe_offset + 1), config.horizon):
        if float(predictions[offset]) > cost:
            returns.append(float(test[TARGET_COLUMN].iloc[offset]) - cost)
    return returns
