from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from integrations.nobitex.ensemble_adapter import (
    EnsembleAdapterConfig,
    NobitexEnsemblePaperAdapter,
)
from integrations.nobitex.mvp import (
    BookLevel,
    LiveExecutionBlocked,
    OrderBookError,
    OrderBookSnapshot,
    synthetic_order_book,
)
from research.ai_ensemble.ensemble import StackingConfig, StackingEnsemble
from research.ai_ensemble.features import (
    FEATURE_COLUMNS,
    build_causal_features,
)
from research.ai_ensemble.hmm import CausalGaussianHMM
from research.ai_ensemble.labels import build_net_profit_targets
from research.ai_ensemble.risk import fractional_kelly_size
from research.ai_ensemble.validation import (
    create_purged_walk_forward_folds,
    mean_return_z_score,
)
from scripts import fetch_nobitex_public_ohlcv as ohlcv_source
from scripts.fetch_nobitex_prehack_ohlcv import (
    EXPECTED_CANDLES as PREHACK_EXPECTED_CANDLES,
    PREHACK_END_TIMESTAMP,
    PREHACK_START_TIMESTAMP,
    fetch_prehack_candles,
)
from scripts.train_nobitex_ai_ensemble import (
    _validate_manifest_row_count,
    _validate_manifest_source_policy,
    run_training,
)


def market_frame(rows: int = 360) -> DataFrame:
    dates = pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="1h", tz="UTC")
    steps = np.arange(rows, dtype=float)
    close = 100.0 + (steps * 0.015) + (1.4 * np.sin(steps / 4.0))
    opened = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(opened, close) + 0.35
    low = np.minimum(opened, close) - 0.35
    return DataFrame(
        {
            "date": dates,
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": 25.0 + (steps % 13),
        }
    )


def test_ohlcv_features_are_causal_and_require_no_external_archives() -> None:
    market = market_frame()
    features = build_causal_features(market)
    prefix = build_causal_features(market.iloc[:280])
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 14
    assert features.iloc[40:].notna().all().all()
    pd.testing.assert_frame_equal(
        features.iloc[:280].reset_index(drop=True),
        prefix.reset_index(drop=True),
    )


@pytest.mark.parametrize("constant_volume", [0.0, 25.0])
def test_volume_zscore_is_neutral_for_constant_volume(
    constant_volume: float,
) -> None:
    market = market_frame()
    market["volume"] = constant_volume

    features = build_causal_features(market)
    volume_zscore = features["volume_zscore_24"]

    # Keep the 24-candle warm-up, then represent absent/constant volume as
    # neutral rather than dropping every otherwise-valid training row.
    assert volume_zscore.iloc[:23].isna().all()
    assert volume_zscore.iloc[23:].notna().all()
    np.testing.assert_array_equal(
        volume_zscore.iloc[23:].to_numpy(dtype=float),
        np.zeros(len(volume_zscore) - 23, dtype=float),
    )
    assert features.iloc[40:].notna().all().all()


def test_ohlcv_validation_rejects_gaps_duplicates_and_missing_columns() -> None:
    market = market_frame()
    gap = market.drop(index=120).reset_index(drop=True)
    with pytest.raises(ValueError, match="contiguous"):
        build_causal_features(gap)
    duplicate = market.copy()
    duplicate.loc[120, "date"] = duplicate.loc[119, "date"]
    with pytest.raises(ValueError, match="unique_and_increasing"):
        build_causal_features(duplicate)
    missing = market.drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing_columns"):
        build_causal_features(missing)


def test_causal_hmm_returns_filtered_three_regime_posteriors() -> None:
    random = np.random.default_rng(71)
    observations = np.column_stack(
        (
            random.normal(0.0, 0.01, 500),
            random.uniform(0.001, 0.03, 500),
            random.uniform(0.001, 0.02, 500),
        )
    )
    model = CausalGaussianHMM(max_iter=8).fit(observations[:400])
    short = model.filter(
        observations[400:450],
        initial_state_probabilities=model.next_prior_probabilities(),
    )
    extended = model.filter(
        observations[400:480],
        initial_state_probabilities=model.next_prior_probabilities(),
    )
    assert short.shape == (50, 3)
    assert np.allclose(short.sum(axis=1), 1.0)
    assert np.allclose(short, extended[:50])
    assert model.means_ is not None
    assert model.state_order_ is not None
    assert np.all(np.diff(model.means_[model.state_order_, 0]) >= 0)


def test_triple_barrier_targets_use_next_open_and_exact_50_bps_cost() -> None:
    market = market_frame(240)
    labels = build_net_profit_targets(
        market.loc[:, ["date", "open", "high", "low", "close", "volume"]]
    )
    valid = labels.dropna(subset=["target", "net_return", "label_end_idx"])
    assert len(valid) > 0
    assert set(valid["label"].unique()).issubset({"BUY", "SELL", "HOLD"})
    valid_indices = valid.index.to_numpy(dtype=int)
    valid_ends = valid["label_end_idx"].to_numpy(dtype=int)
    assert np.all(valid_ends >= valid_indices)
    with pytest.raises(ValueError, match="must_be_50"):
        build_net_profit_targets(
            market.loc[:, ["date", "open", "high", "low", "close", "volume"]],
            round_trip_cost_bps=30,
        )


def test_purged_walk_forward_and_dsr_use_non_overlapping_trade_samples() -> None:
    count = 600
    eligible = list(range(count))
    ends = [index + 24 if index + 24 < count else -1 for index in range(count)]
    folds = create_purged_walk_forward_folds(
        eligible,
        ends,
        fold_count=3,
        first_validation_fraction=0.60,
        embargo_bars=24,
        minimum_train_samples=100,
    )
    assert len(folds) == 3
    for fold in folds:
        validation_start = fold.validation_indices[0]
        assert max(ends[index] for index in fold.train_indices) < validation_start
        assert max(fold.train_indices) < validation_start - 24
    assert mean_return_z_score([0.02, -0.01, 0.03, 0.01]) is not None
    assert mean_return_z_score([0.01]) is None
    assert mean_return_z_score([0.01, 0.01]) is None


def test_fractional_kelly_blocks_cost_negative_edge_and_caps_position() -> None:
    blocked = fractional_kelly_size(
        probability_of_net_profit=0.8,
        equity=Decimal("10000"),
        reference_price=Decimal("1000"),
        atr=Decimal("0.1"),
    )
    assert not blocked.approved
    assert blocked.quote_notional == Decimal("0")
    sized = fractional_kelly_size(
        probability_of_net_profit=0.8,
        equity=Decimal("10000"),
        reference_price=Decimal("1000"),
        atr=Decimal("10"),
    )
    assert sized.approved
    assert sized.position_fraction <= 0.10
    assert sized.quote_notional <= Decimal("1000")


class FixedProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features: DataFrame) -> np.ndarray:
        return np.full(len(features), self.probability, dtype=float)


def test_nobitex_adapter_returns_only_a_liquidity_gated_paper_plan() -> None:
    adapter = NobitexEnsemblePaperAdapter(FixedProbabilityModel(0.8))
    decision = adapter.plan_entry(
        DataFrame({"feature": [0.1]}),
        snapshot=synthetic_order_book(1000.0, "BTCUSDT"),
        equity=Decimal("10000"),
        atr=Decimal("10"),
    )
    assert decision.status == "PAPER_PLAN_ONLY"
    assert decision.maker_plan is not None
    assert decision.maker_plan.execution_permission is False
    assert decision.order_permission is False
    assert decision.private_api_calls == 0
    assert decision.real_order_attempts == 0
    assert all(
        child.post_only_required and not child.exchange_enforced_post_only_confirmed
        for child in decision.maker_plan.child_slices
    )


def test_nobitex_adapter_fail_closed_branches() -> None:
    book = synthetic_order_book(1000.0, "BTCUSDT")
    low_probability = NobitexEnsemblePaperAdapter(
        FixedProbabilityModel(0.4),
        EnsembleAdapterConfig(minimum_probability=0.6),
    )
    rejected = low_probability.plan_entry(
        DataFrame({"feature": [0.1]}),
        snapshot=book,
        equity=Decimal("10000"),
        atr=Decimal("10"),
    )
    assert rejected.status == "NO_SIGNAL"
    assert rejected.order_permission is False
    with pytest.raises(LiveExecutionBlocked):
        NobitexEnsemblePaperAdapter(
            FixedProbabilityModel(0.8),
            EnsembleAdapterConfig(dry_run=False),
        )
    with pytest.raises(ValueError, match="symbol_invalid"):
        EnsembleAdapterConfig(symbol="BTC/USDT").validate()
    with pytest.raises(ValueError, match="50_bps"):
        EnsembleAdapterConfig(round_trip_cost_bps=30).validate()
    with pytest.raises(ValueError, match="fixed_by_owner"):
        EnsembleAdapterConfig(band=Decimal("0.01")).validate()
    with pytest.raises(OrderBookError, match="symbol_mismatch"):
        NobitexEnsemblePaperAdapter(FixedProbabilityModel(0.8)).plan_entry(
            DataFrame({"feature": [0.1]}),
            snapshot=replace(book, symbol="ETHUSDT"),
            equity=Decimal("10000"),
            atr=Decimal("10"),
        )
    stale = replace(
        book,
        last_update=datetime.now(timezone.utc) - timedelta(seconds=10),
        received_at=datetime.now(timezone.utc),
    )
    with pytest.raises(OrderBookError, match="stale"):
        NobitexEnsemblePaperAdapter(FixedProbabilityModel(0.8)).plan_entry(
            DataFrame({"feature": [0.1]}),
            snapshot=stale,
            equity=Decimal("10000"),
            atr=Decimal("10"),
        )
    tiny_book = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update=datetime.now(timezone.utc),
        received_at=datetime.now(timezone.utc),
        bids=(BookLevel(Decimal("999"), Decimal("0.001")),),
        asks=(BookLevel(Decimal("1001"), Decimal("0.001")),),
    )
    liquidity_rejected = NobitexEnsemblePaperAdapter(
        FixedProbabilityModel(0.8)
    ).plan_entry(
        DataFrame({"feature": [0.1]}),
        snapshot=tiny_book,
        equity=Decimal("10000"),
        atr=Decimal("10"),
    )
    assert liquidity_rejected.status == "LIQUIDITY_REJECTED"
    risk_rejected = NobitexEnsemblePaperAdapter(FixedProbabilityModel(0.8)).plan_entry(
        DataFrame({"feature": [0.1]}),
        snapshot=book,
        equity=Decimal("10000"),
        atr=Decimal("0.1"),
    )
    assert risk_rejected.status == "RISK_REJECTED"
    empty_features = NobitexEnsemblePaperAdapter(FixedProbabilityModel(0.8))
    with pytest.raises(ValueError, match="features_empty"):
        empty_features.plan_entry(
            DataFrame(),
            snapshot=book,
            equity=Decimal("10000"),
            atr=Decimal("10"),
        )
    invalid_model = NobitexEnsemblePaperAdapter(FixedProbabilityModel(1.5))
    with pytest.raises(ValueError, match="probability_invalid"):
        invalid_model.plan_entry(
            DataFrame({"feature": [0.1]}),
            snapshot=book,
            equity=Decimal("10000"),
            atr=Decimal("10"),
        )


def test_stacking_model_contract_and_training_smoke_when_dependencies_exist() -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    random = np.random.default_rng(19)
    count = 720
    returns = random.normal(0.0, 0.01, count)
    features = DataFrame(
        {
            "log_return_1": returns,
            "realized_volatility_24": np.abs(random.normal(0.01, 0.002, count)),
            "range_fraction": np.abs(random.normal(0.02, 0.003, count)),
            "signal_a": returns + random.normal(0, 0.01, count),
            "signal_b": random.normal(0, 1, count),
        }
    )
    labels = (features["log_return_1"].to_numpy(dtype=float) > 0).astype(int)
    end_indices = [index + 24 if index + 24 < count else -1 for index in range(count)]
    training_indices = [index for index, end in enumerate(end_indices) if end >= 0]
    config = StackingConfig(
        fold_count=2,
        embargo_bars=24,
        minimum_train_samples=60,
        n_estimators=10,
        probability_threshold=0.5,
    )
    model = StackingEnsemble(config).fit(
        features,
        labels.tolist(),
        end_indices,
        training_indices,
    )
    output = model.predict_proba(features.iloc[-10:])
    assert output.shape == (10,)
    assert np.all((output >= 0.0) & (output <= 1.0))


def test_prehack_fetcher_uses_exact_nobitex_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_fetch(**kwargs: Any) -> list[ohlcv_source.CandleRecord]:
        captured.update(kwargs)
        return [
            {
                "timestamp": stamp,
                "date": datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            }
            for stamp in range(PREHACK_START_TIMESTAMP, PREHACK_END_TIMESTAMP + 1, 3600)
        ]

    monkeypatch.setattr(ohlcv_source, "fetch_candles_between", fake_fetch)
    rows = fetch_prehack_candles(
        request_json=lambda _: {},
        sleep_fn=lambda _: None,
    )
    assert captured["start_timestamp"] == PREHACK_START_TIMESTAMP
    assert captured["end_timestamp"] == PREHACK_END_TIMESTAMP
    assert captured["interval"] == "1h"
    assert captured["symbol"] == "BTCUSDT"
    assert len(rows) == PREHACK_EXPECTED_CANDLES


class FakeChronosPipeline:
    """Deterministic Chronos API double for unit tests only."""

    def predict_df(
        self,
        context: DataFrame,
        *,
        prediction_length: int,
        id_column: str,
        timestamp_column: str,
        target: str,
        **kwargs: Any,
    ) -> DataFrame:
        predictions: list[dict[str, Any]] = []
        for item_id, group in context.groupby(id_column, sort=False):
            value = float(group[target].iloc[-1])
            timestamp = pd.Timestamp(group[timestamp_column].iloc[-1])
            for step in range(1, prediction_length + 1):
                predictions.append(
                    {
                        id_column: item_id,
                        timestamp_column: timestamp + pd.Timedelta(hours=step),
                        "predictions": value + 0.0001 * step,
                    }
                )
        return DataFrame.from_records(predictions)


def test_end_to_end_training_writes_unpromoted_ohlcv_artifacts(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    market = market_frame(1200)
    ohlcv_path = tmp_path / "ohlcv.csv"
    manifest_path = tmp_path / "ohlcv.manifest.json"
    model_path = tmp_path / "ensemble.joblib"
    model_manifest_path = tmp_path / "ensemble.manifest.json"
    report_path = tmp_path / "report.json"
    ohlcv = market.loc[:, ["date", "open", "high", "low", "close", "volume"]]
    ohlcv.to_csv(ohlcv_path, index=False)
    data_manifest = {
        "schema_version": "alix.nobitex.public-ohlcv.v2",
        "provider": "TEST_FIXTURE",
        "symbol": "BTCUSDT",
        "pair": "BTC/USDT",
        "interval": "1h",
        "requested_days": 50,
        "row_count": len(ohlcv),
        "missing_intervals": 0,
        "start_utc": pd.Timestamp(ohlcv["date"].iloc[0]).isoformat(),
        "end_utc": pd.Timestamp(ohlcv["date"].iloc[-1]).isoformat(),
        "csv_sha256": sha256(ohlcv_path.read_bytes()).hexdigest(),
        "commit_sha": "test-commit",
        "public_api_only": False,
        "credentials_used": False,
        "private_api_calls": 0,
        "exchange_calls": 0,
        "real_order_attempts": 0,
        "order_permission": False,
        "Champion": None,
        "trading_authorization": "NO-GO",
        "data_authority": "SYNTHETIC_TEST_FIXTURE_NOT_ECONOMIC_EVIDENCE",
    }
    manifest_path.write_text(
        json.dumps(data_manifest, sort_keys=True), encoding="utf-8"
    )
    report = run_training(
        ohlcv_path=ohlcv_path,
        manifest_path=manifest_path,
        model_path=model_path,
        model_manifest_path=model_manifest_path,
        report_path=report_path,
        commit_sha="test-commit",
        chronos_pipeline=FakeChronosPipeline(),
    )
    artifact = joblib.load(model_path)
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == "alix.nobitex.ai-ensemble.v2"
    assert "chronos_pred" in artifact["feature_columns"]
    assert artifact["round_trip_cost_bps"] == 50
    assert artifact["dataset_csv_sha256"] == data_manifest["csv_sha256"]
    assert report["dataset"]["row_count"] == 1200
    assert report["dataset"]["eligible_labeled_rows"] >= 500
    assert report["dataset"]["kind"] == "NON_PRODUCTION_TEST_FIXTURE"
    assert report["validation"]["fold_count"] == 3
    assert report["research_result"]["statistical_z_score"] is not None
    assert "net_profit_factor" in report["baseline_research_result"]
    assert "net_profit_factor" in report["candidate_research_result"]
    assert report["chronos"]["forecast_type"] == "zero_shot"
    assert "chronos_pred" in report["model"]["candidate_feature_columns"]
    assert "cost_sensitivity_90_bps" in report
    assert report["artifact"]["sha256"] == model_manifest["artifact"]["sha256"]
    assert report["safety"]["Champion"] is None
    assert report["safety"]["order_permission"] is False
    assert report["safety"]["private_api_calls"] == 0
    assert report["safety"]["real_order_attempts"] == 0
    assert report["safety"]["trading_authorization"] == "NO-GO"
    assert report_path.is_file()


def test_manifest_row_count_accepts_exact_inclusive_prehack_count() -> None:
    manifest = {
        "interval": "1h",
        "row_count": 31_801,
        "requested_days": 1_325,
        "expected_candles": 31_801,
        "missing_intervals": 0,
    }
    ohlcv = DataFrame(index=range(31_801))
    assert _validate_manifest_row_count(manifest, ohlcv) == (31_801, 1_325)


def test_manifest_row_count_rejects_off_by_one_expected_count() -> None:
    manifest = {
        "interval": "1h",
        "row_count": 31_801,
        "requested_days": 1_325,
        "expected_candles": 31_800,
        "missing_intervals": 0,
    }
    with pytest.raises(
        ValueError,
        match="research_manifest_expected_candles_mismatch",
    ):
        _validate_manifest_row_count(manifest, DataFrame(index=range(31_801)))


def test_public_dataset_manifest_must_be_nobitex_only_and_exact_window() -> None:
    timestamps = np.arange(
        PREHACK_START_TIMESTAMP,
        PREHACK_END_TIMESTAMP + 3600,
        3600,
        dtype=np.int64,
    )
    ohlcv = DataFrame({"date": pd.to_datetime(timestamps, unit="s", utc=True)})
    manifest = {
        "public_api_only": True,
        "source_policy": "NOBITEX_ONLY_NO_BACKFILL",
        "data_sources": ["Nobitex public UDF history API"],
        "dataset_window": "PRE_HACK_NOBITEX_ONLY",
        "start_timestamp": PREHACK_START_TIMESTAMP,
        "end_timestamp": PREHACK_END_TIMESTAMP,
        "expected_candles": PREHACK_EXPECTED_CANDLES,
        "requested_days": 1325,
    }
    _validate_manifest_source_policy(manifest, ohlcv)


def test_public_dataset_manifest_rejects_external_source_override() -> None:
    timestamps = np.arange(
        PREHACK_START_TIMESTAMP,
        PREHACK_END_TIMESTAMP + 3600,
        3600,
        dtype=np.int64,
    )
    ohlcv = DataFrame({"date": pd.to_datetime(timestamps, unit="s", utc=True)})
    manifest = {
        "public_api_only": True,
        "source_policy": "NOBITEX_ONLY_NO_BACKFILL",
        "data_sources": ["Nobitex public UDF history API"],
        "source_overrides": ["Binance"],
        "dataset_window": "PRE_HACK_NOBITEX_ONLY",
        "start_timestamp": PREHACK_START_TIMESTAMP,
        "end_timestamp": PREHACK_END_TIMESTAMP,
        "expected_candles": PREHACK_EXPECTED_CANDLES,
        "requested_days": 1325,
    }
    with pytest.raises(
        ValueError,
        match="research_manifest_external_source_override_forbidden",
    ):
        _validate_manifest_source_policy(manifest, ohlcv)
