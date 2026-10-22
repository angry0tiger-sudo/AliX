from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from services.freqai_baseline.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_market_features,
    forward_return_target,
)
from services.freqai_baseline.walkforward import (
    WalkForwardConfig,
    make_walk_forward_folds,
    run_walk_forward,
    validate_ohlcv,
)
from user_data.strategies.AliXFreqAIBaselineStrategy import AliXFreqAIBaselineStrategy


class MeanRegressor:
    """Small deterministic estimator used only to test evaluation mechanics."""

    def __init__(self, **kwargs: Any) -> None:
        self.parameters = kwargs
        self.mean_ = 0.0

    def fit(self, features: DataFrame, labels: np.ndarray) -> MeanRegressor:
        self.mean_ = float(np.mean(labels))
        return self

    def predict(self, features: DataFrame) -> np.ndarray:
        return np.full(len(features), self.mean_, dtype=float)


class FakeFreqAI:
    def __init__(self) -> None:
        self.called = 0

    def start(
        self, dataframe: DataFrame, metadata: dict[str, Any], strategy: Any
    ) -> DataFrame:
        self.called += 1
        result = dataframe.copy()
        result["do_predict"] = 1
        return result


def market_frame(rows: int = 620) -> DataFrame:
    rng = np.random.default_rng(223344)
    dates = pd.date_range("2020-01-01T00:00:00Z", periods=rows, freq="h")
    close = np.empty(rows, dtype=float)
    opened = np.empty(rows, dtype=float)
    close[0] = 100.0
    opened[0] = 100.0
    for index in range(1, rows):
        opened[index] = close[index - 1]
        regime = 0.003 if (index // 100) % 2 == 0 else 0.01
        close[index] = opened[index] * np.exp(
            0.00003 + regime * rng.standard_normal()
        )
    high = np.maximum(opened, close) * 1.004
    low = np.minimum(opened, close) * 0.996
    volume = rng.lognormal(mean=4.0, sigma=0.3, size=rows)
    return DataFrame(
        {
            "date": dates,
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_forward_target_is_future_label_not_feature() -> None:
    frame = market_frame(40)
    target = forward_return_target(frame, horizon=3)
    expected = frame["close"].shift(-3) / frame["close"] - 1.0

    assert target.name == TARGET_COLUMN
    pd.testing.assert_series_equal(target, expected.rename(TARGET_COLUMN))
    assert target.iloc[-3:].isna().all()


def test_feature_values_at_time_t_do_not_depend_on_later_rows() -> None:
    frame = market_frame(80)
    original = add_market_features(frame)
    changed = frame.copy()
    changed.loc[40:, ["open", "high", "low", "close", "volume"]] *= 7.0
    mutated = add_market_features(changed)

    for column in FEATURE_COLUMNS:
        np.testing.assert_allclose(
            original.loc[:35, column].to_numpy(),
            mutated.loc[:35, column].to_numpy(),
            equal_nan=True,
        )


def test_forward_target_rejects_invalid_horizon() -> None:
    frame = market_frame(20)
    for invalid in (0, -1, True):
        try:
            forward_return_target(frame, horizon=invalid)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError("invalid target horizon should be rejected")


def test_walk_forward_folds_purge_target_horizon_before_test_start() -> None:
    config = WalkForwardConfig(horizon=4, n_splits=3, min_train_size=60)
    folds = make_walk_forward_folds(400, config)

    assert len(folds) == 3
    for fold in folds:
        assert fold.train_stop + config.horizon <= fold.test_start
        assert fold.test_start < fold.test_stop
        assert fold.train_start == 0


def test_walk_forward_rejects_invalid_configuration_and_insufficient_data() -> None:
    for kwargs in (
        {"horizon": 0},
        {"n_splits": 1},
        {"min_train_size": 10},
        {"fee_bps_per_side": -1.0},
        {"slippage_bps_per_side": -1.0},
    ):
        try:
            WalkForwardConfig(**kwargs)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError("invalid walk-forward configuration should fail")
    try:
        make_walk_forward_folds(
            90, WalkForwardConfig(horizon=12, n_splits=3, min_train_size=300)
        )
    except ValueError as exc:
        assert "insufficient valid rows" in str(exc)
    else:
        raise AssertionError("insufficient data should fail closed")


def test_ohlcv_gate_rejects_gap_missing_fields_duplicate_columns_and_bad_timestamp() -> None:
    frame = market_frame(5)
    gap = frame.copy()
    gap.loc[3, "date"] = gap.loc[2, "date"] + pd.Timedelta(hours=2)
    try:
        validate_ohlcv(gap, timeframe_seconds=3600)
    except ValueError as exc:
        assert "gap_detected" in str(exc)
    else:
        raise AssertionError("gap should be rejected")
    try:
        validate_ohlcv(frame.drop(columns=["volume"]), timeframe_seconds=3600)
    except ValueError as exc:
        assert "volume" in str(exc)
    else:
        raise AssertionError("missing volume should be rejected")
    duplicated = frame.copy()
    duplicated["close_duplicate"] = duplicated["close"]
    duplicated.columns = list(duplicated.columns[:-1]) + ["close"]
    try:
        validate_ohlcv(duplicated, timeframe_seconds=3600)
    except ValueError as exc:
        assert "duplicate column names" in str(exc)
    else:
        raise AssertionError("duplicate columns should be rejected")
    naive = frame.copy()
    naive["date"] = naive["date"].dt.tz_localize(None)
    try:
        validate_ohlcv(naive, timeframe_seconds=3600)
    except ValueError as exc:
        assert "timezone_required" in str(exc)
    else:
        raise AssertionError("naive timestamps should be rejected")


def test_walk_forward_report_reloads_model_and_never_promotes_champion() -> None:
    frame = market_frame(620)
    config = WalkForwardConfig(
        horizon=3,
        n_splits=2,
        min_train_size=60,
        fee_bps_per_side=10.0,
        slippage_bps_per_side=5.0,
    )
    report = run_walk_forward(
        frame,
        pair="FIXTURE/USDT",
        timeframe="1h",
        timeframe_seconds=3600,
        config=config,
        dataset_kind="synthetic",
        dataset_sha256="a" * 64,
        commit_sha="test-commit",
        estimator_factory=MeanRegressor,
    )

    assert report["evaluation_status"] == "SYNTHETIC_MECHANICS_ONLY"
    assert report["model"]["reload_roundtrip"] == "passed"
    assert report["model"]["roundtrip_prediction_max_abs_diff"] == 0.0
    assert report["out_of_sample"]["sample_count"] > 0
    assert report["out_of_sample"]["forecast"]["rmse"] >= 0.0
    assert len(report["folds"]) == 2
    assert all(fold["purged_label_rows"] == 3 for fold in report["folds"])
    assert report["out_of_sample"]["costed_fixed_horizon_research_rule"][
        "no_trade_net_return"
    ] == 0.0
    assert report["economic_gates"]["status"] == "NOT_EVALUATED"
    assert report["champion"] is None
    assert report["promotion_status"] == "BLOCKED_NO_ECONOMIC_GATE"


def test_freqai_strategy_keeps_all_entry_and_exit_signals_closed() -> None:
    strategy = AliXFreqAIBaselineStrategy()
    fake_freqai = FakeFreqAI()
    setattr(strategy, "freqai", fake_freqai)
    valid = market_frame(3).iloc[:2].copy()
    checked = strategy.populate_indicators(valid, {"pair": "FIXTURE/USDT"})
    assert fake_freqai.called == 1
    assert checked["alix_data_quality_valid"].all()

    entries = strategy.populate_entry_trend(checked.copy(), {"pair": "FIXTURE/USDT"})
    exits = strategy.populate_exit_trend(checked.copy(), {"pair": "FIXTURE/USDT"})
    assert entries["enter_long"].eq(0).all()
    assert exits["exit_long"].eq(0).all()

    gap = market_frame(3).iloc[[0, 2]].copy()
    rejected = strategy.populate_indicators(gap, {"pair": "FIXTURE/USDT"})
    assert fake_freqai.called == 1
    assert not rejected["alix_data_quality_valid"].any()
    assert rejected["do_predict"].eq(0).all()


def test_freqai_target_honors_versioned_horizon_contract() -> None:
    strategy = AliXFreqAIBaselineStrategy()
    setattr(strategy, "freqai_info", {"feature_parameters": {"label_period_candles": 5}})
    frame = market_frame(25)
    actual = strategy.set_freqai_targets(frame.copy(), {"pair": "FIXTURE/USDT"})
    expected = frame["close"].shift(-5) / frame["close"] - 1.0

    pd.testing.assert_series_equal(actual[TARGET_COLUMN], expected.rename(TARGET_COLUMN))
