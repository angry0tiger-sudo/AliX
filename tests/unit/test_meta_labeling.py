"""Contract tests for offline Triple-Barrier and OOF-only Meta-Labeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from services.freqai_baseline.features import (
    FEATURE_COLUMNS,
    add_market_features,
    forward_return_target,
)
from services.data_quality.gate import validate_candles
from services.meta_labeling.meta_model import (
    META_MODEL_SCHEMA_VERSION,
    MetaLabelModel,
    OOFFold,
    generate_oof_primary_predictions,
)
from services.meta_labeling.triple_barrier import (
    TripleBarrierConfig,
    apply_triple_barrier,
)
from services.meta_labeling.walkforward import evaluate_meta_oof


def flat_candles(rows: int = 40) -> DataFrame:
    dates = pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="h")
    close = np.full(rows, 100.0)
    return DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(rows, 1000.0),
        }
    )


def event_frame(
    event_index: int = 5,
    side: int = 1,
    volatility: float = 0.01,
) -> DataFrame:
    return DataFrame(
        {
            "event_index": [event_index],
            "side": [side],
            "target_volatility": [volatility],
        }
    )


def short_market(rows: int = 320, seed: int = 505) -> DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="h")
    close = np.empty(rows, dtype=float)
    opened = np.empty(rows, dtype=float)
    close[0] = 100.0
    opened[0] = 100.0
    for index in range(1, rows):
        regime = 0.0025 if (index // 35) % 2 == 0 else 0.007
        opened[index] = close[index - 1]
        close[index] = opened[index] * np.exp(0.00005 + regime * rng.standard_normal())
    range_width = rng.uniform(0.0005, 0.004, size=rows)
    high = np.maximum(opened, close) * (1.0 + range_width)
    low = np.minimum(opened, close) * (1.0 - range_width)
    return DataFrame(
        {
            "date": dates,
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.lognormal(mean=4.0, sigma=0.2, size=rows),
        }
    )


def test_triple_barrier_take_profit_and_costed_meta_label() -> None:
    candles = flat_candles()
    candles.loc[6, "high"] = 101.0
    result = apply_triple_barrier(
        candles,
        event_frame(),
        TripleBarrierConfig(
            profit_taking_multiplier=0.5,
            stop_loss_multiplier=0.5,
            max_holding_bars=5,
            fee_bps_per_side=10,
            slippage_bps_per_side=5,
        ),
    )
    row = result.iloc[0]
    assert row["exit_reason"] == "take_profit"
    assert row["barrier_label"] == 1
    assert row["label_end_index"] == 6
    assert row["net_return"] > 0
    assert row["meta_label"] == 1
    assert "close" not in result.columns
    assert "future_feature" not in result.columns


def test_triple_barrier_stop_loss_wins_same_candle_collision() -> None:
    candles = flat_candles()
    candles.loc[6, "high"] = 101.0
    candles.loc[6, "low"] = 99.0
    result = apply_triple_barrier(
        candles,
        event_frame(),
        TripleBarrierConfig(
            profit_taking_multiplier=0.5,
            stop_loss_multiplier=0.5,
            max_holding_bars=5,
        ),
    )
    assert result.iloc[0]["exit_reason"] == "stop_loss"
    assert result.iloc[0]["barrier_label"] == -1
    assert result.iloc[0]["meta_label"] == 0


def test_triple_barrier_supports_short_side_and_vertical_expiry() -> None:
    candles = flat_candles()
    candles.loc[6, "low"] = 99.0
    short_result = apply_triple_barrier(
        candles,
        event_frame(side=-1),
        TripleBarrierConfig(
            profit_taking_multiplier=0.5,
            stop_loss_multiplier=0.5,
            max_holding_bars=5,
            fee_bps_per_side=0,
            slippage_bps_per_side=0,
        ),
    )
    assert short_result.iloc[0]["exit_reason"] == "take_profit"
    assert short_result.iloc[0]["gross_return"] > 0
    vertical = apply_triple_barrier(
        flat_candles(),
        event_frame(),
        TripleBarrierConfig(
            profit_taking_multiplier=0.5,
            stop_loss_multiplier=0.5,
            max_holding_bars=5,
        ),
    )
    assert vertical.iloc[0]["exit_reason"] == "vertical"
    assert vertical.iloc[0]["barrier_label"] == 0
    assert vertical.iloc[0]["meta_label"] == 0
    assert vertical.iloc[0]["net_return"] < 0


def test_triple_barrier_fails_closed_on_bad_time_data_and_events() -> None:
    config = TripleBarrierConfig(max_holding_bars=5)
    naive = flat_candles()
    naive["date"] = naive["date"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone_required"):
        apply_triple_barrier(naive, event_frame(), config)

    gap = flat_candles()
    gap.loc[7, "date"] = pd.Timestamp("2024-01-01T07:30:00Z")
    with pytest.raises(ValueError, match="gap_detected"):
        apply_triple_barrier(gap, event_frame(), config)

    invalid_side = event_frame(side=0)
    with pytest.raises(ValueError, match="side_must"):
        apply_triple_barrier(flat_candles(), invalid_side, config)
    invalid_volatility = event_frame(volatility=float("nan"))
    with pytest.raises(ValueError, match="target_volatility"):
        apply_triple_barrier(flat_candles(), invalid_volatility, config)
    outside = event_frame(event_index=39)
    with pytest.raises(ValueError, match="event_horizon_out_of_bounds"):
        apply_triple_barrier(flat_candles(), outside, config)
    duplicate = pd.concat([event_frame(), event_frame()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate_event_index"):
        apply_triple_barrier(flat_candles(), duplicate, config)
    with pytest.raises(ValueError, match="barrier multipliers"):
        TripleBarrierConfig(profit_taking_multiplier=0)
    with pytest.raises(ValueError, match="fee and slippage"):
        TripleBarrierConfig(fee_bps_per_side=-1)


def test_shared_data_quality_gate_rejects_bad_synthetic_candles() -> None:
    candles = short_market(rows=320)
    result = validate_candles(
        [
            {str(key): value for key, value in row.items()}
            for row in candles.to_dict(orient="records")
        ],
        expected_interval_seconds=3600,
        gap_policy="reject",
    )
    assert result.passed is True
    assert result.status is not None
    assert result.status.value == "VALID"
    assert result.fingerprint

    gap = candles.copy()
    gap.loc[7, "date"] = pd.Timestamp("2024-01-01T07:30:00Z")
    rejected = validate_candles(
        [
            {str(key): value for key, value in row.items()}
            for row in gap.to_dict(orient="records")
        ],
        expected_interval_seconds=3600,
        gap_policy="reject",
    )
    assert rejected.passed is False
    assert "gap_detected" in ",".join(rejected.reasons)


def test_oof_primary_predictions_are_purged_and_never_in_sample() -> None:
    rng = np.random.default_rng(44)
    features = DataFrame(
        {
            "signal": rng.normal(size=300),
            "noise": rng.normal(size=300),
        }
    )
    target = pd.Series(
        (features["signal"] + 0.2 * rng.normal(size=300) > 0).astype(float)
    )
    label_end = np.arange(len(features), dtype=int) + 1
    folds = [
        OOFFold(0, 100, 100, 150),
        OOFFold(0, 150, 150, 200),
        OOFFold(0, 200, 200, 250),
    ]
    oof = generate_oof_primary_predictions(
        features,
        target,
        label_end,
        folds,
        lambda: LogisticRegression(random_state=42, max_iter=200),
    )
    assert oof["event_index"].is_unique
    assert set(oof["fold_id"].unique()) == {0, 1, 2}
    assert oof["event_index"].min() >= 100
    assert oof["event_index"].max() < 250
    assert (oof["primary_train_max_index"] < oof["event_index"]).all()
    assert (oof["primary_train_max_label_end_index"] < oof["event_index"]).all()
    assert (oof["primary_train_count"] > 2).all()


def test_meta_model_requires_oof_and_mature_labels_and_roundtrips_json() -> None:
    rng = np.random.default_rng(77)
    event_indices = np.arange(100, 180, dtype=int)
    features = DataFrame(
        {
            "feature_a": rng.normal(size=len(event_indices)),
            "feature_b": rng.normal(size=len(event_indices)),
            "primary_prediction": np.where(event_indices % 3 == 0, 1.0, 0.0),
        },
        index=event_indices,
    )
    labels = (features["feature_a"] + 0.4 * features["feature_b"] > 0).astype(int)
    provenance = DataFrame(
        {
            "event_index": event_indices,
            "fold_id": np.ones(len(event_indices), dtype=int),
            "primary_train_max_index": event_indices - 4,
            "primary_train_max_label_end_index": event_indices - 2,
        }
    )
    label_ends = event_indices + 1
    model = MetaLabelModel().fit(
        features,
        labels.to_numpy(dtype=int),
        provenance,
        label_ends,
        evaluation_cutoff_index=250,
    )
    assert model.fitted
    assert len(model.predict_probability(features)) == len(features)
    assert set(np.unique(labels)) == {0, 1}
    loaded = MetaLabelModel.from_json(model.to_json())
    assert loaded.to_dict()["schema_version"] == META_MODEL_SCHEMA_VERSION
    np.testing.assert_allclose(
        model.predict_probability(features),
        loaded.predict_probability(features),
        atol=1e-12,
        rtol=0.0,
    )
    filtered = loaded.filter_research_candidates(features, minimum_probability=0.5)
    assert filtered["research_filter_pass"].dtype == bool
    assert "enter_long" not in filtered.columns
    assert "exit_long" not in filtered.columns
    assert "risk_approved" not in filtered.columns
    assert set(filtered["reason_code"].unique()).issubset(
        {"score_above_research_cutoff", "score_below_research_cutoff"}
    )


def test_meta_model_fail_closed_and_rejects_leaked_provenance() -> None:
    features = DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}, index=[10, 11, 12, 13])
    unfitted = MetaLabelModel().filter_research_candidates(features, 0.5)
    assert not unfitted["research_filter_pass"].any()
    assert unfitted["reason_code"].eq("meta_model_or_input_invalid_fail_closed").all()
    invalid_threshold = MetaLabelModel().filter_research_candidates(
        features, float("nan")
    )
    assert not invalid_threshold["research_filter_pass"].any()

    labels = np.array([0, 1, 0, 1], dtype=int)
    provenance = DataFrame(
        {
            "event_index": [10, 11, 12, 13],
            "fold_id": [0, 0, 0, 0],
            "primary_train_max_index": [5, 6, 7, 13],
            "primary_train_max_label_end_index": [6, 7, 8, 12],
        }
    )
    with pytest.raises(ValueError, match="oof_or_purge_contract_failed"):
        MetaLabelModel().fit(
            features, labels, provenance, [7, 8, 9, 10], evaluation_cutoff_index=20
        )
    provenance.loc[3, "primary_train_max_index"] = 9
    provenance.loc[3, "primary_train_max_label_end_index"] = 8
    with pytest.raises(ValueError, match="oof_or_purge_contract_failed"):
        MetaLabelModel().fit(
            features, labels, provenance, [7, 8, 9, 20], evaluation_cutoff_index=20
        )
    with pytest.raises(ValueError, match="schema_version_mismatch"):
        MetaLabelModel.from_dict({"schema_version": "old"})
    with pytest.raises(ValueError, match="cannot_serialize_unfitted"):
        MetaLabelModel().to_json()


def test_integrated_meta_walkforward_purges_maturing_barrier_labels() -> None:
    candles = short_market()
    engineered = add_market_features(candles)
    features = engineered.loc[:, list(FEATURE_COLUMNS)]
    forward = forward_return_target(candles, horizon=1).to_numpy(dtype=float)
    target = pd.Series(
        np.where(np.isfinite(forward), (forward > 0.0).astype(float), np.nan)
    )
    primary_ends = np.arange(len(candles), dtype=int) + 1
    folds = [
        OOFFold(0, 100, 100, 150),
        OOFFold(0, 150, 150, 200),
        OOFFold(0, 200, 200, 250),
    ]
    oof = generate_oof_primary_predictions(
        features,
        target,
        primary_ends,
        folds,
        lambda: LogisticRegression(random_state=42, max_iter=300),
    )
    indices = oof["event_index"].to_numpy(dtype=int)
    volatility = features["%-volatility_12"].iloc[indices].to_numpy(dtype=float)
    events = DataFrame(
        {
            "event_index": indices,
            "side": np.where(
                oof["primary_prediction"].to_numpy(dtype=float) >= 0.5, 1, -1
            ),
            "target_volatility": volatility,
        }
    )
    labels = apply_triple_barrier(
        candles,
        events,
        TripleBarrierConfig(max_holding_bars=5, timeframe_seconds=3600),
    )
    report = evaluate_meta_oof(features, oof, labels, minimum_probability=0.5)
    assert report["training_signal_requirement"] == "OUT_OF_FOLD_ONLY"
    assert report["economic_gates"]["status"] == "NOT_EVALUATED"
    assert report["champion"] is None
    assert report["promotion_status"] == "BLOCKED_NO_ECONOMIC_GATE"
    assert report["trading_authorization"] == "NO_GO"
    assert report["model_output_is_risk_approval"] is False
    assert report["evaluated_fold_count"] >= 1
    for fold in report["folds"]:
        if fold["status"] == "EVALUATED":
            assert fold["train_max_label_end_index"] < fold["test_start_index"]
            assert fold["meta_model_reload_roundtrip"] == "passed"
            assert fold["meta_model_reload_max_abs_diff"] == 0.0
