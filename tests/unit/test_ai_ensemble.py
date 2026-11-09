from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

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
    merge_market_archives,
)
from research.ai_ensemble.hmm import CausalGaussianHMM
from research.ai_ensemble.labels import build_net_profit_targets
from research.ai_ensemble.risk import fractional_kelly_size
from research.ai_ensemble.validation import (
    create_purged_walk_forward_folds,
    deflated_sharpe_ratio,
)


def market_frame(rows: int = 360) -> DataFrame:
    dates = pd.date_range(
        "2024-01-01T00:00:00Z", periods=rows, freq="1h", tz="UTC"
    )
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
            "order_book_imbalance": np.sin(steps / 9.0) * 0.4,
            "taker_buy_volume": 10.0 + (steps % 7),
            "taker_sell_volume": 8.0 + (steps % 5),
            "funding_rate": np.sin(steps / 80.0) * 0.0001,
            "microstructure_timestamp": dates + pd.Timedelta(minutes=30),
            "funding_timestamp": dates - pd.Timedelta(hours=1),
        }
    )


def test_market_archive_requires_exact_one_to_one_timestamps() -> None:
    market = market_frame()
    merged = merge_market_archives(
        market.loc[:, ["date", "open", "high", "low", "close", "volume"]],
        market.loc[
            :,
            [
                "date",
                "order_book_imbalance",
                "taker_buy_volume",
                "taker_sell_volume",
                "funding_rate",
                "microstructure_timestamp",
                "funding_timestamp",
            ],
        ],
    )
    assert len(merged) == len(market)
    broken = market.iloc[1:].copy()
    with pytest.raises(ValueError, match="exactly_match"):
        merge_market_archives(
            market.loc[:, ["date", "open", "high", "low", "close", "volume"]],
            broken.loc[
                :,
                [
                    "date",
                    "order_book_imbalance",
                    "taker_buy_volume",
                    "taker_sell_volume",
                    "funding_rate",
                    "microstructure_timestamp",
                    "funding_timestamp",
                ],
            ],
        )


def test_feature_builder_preserves_microstructure_and_has_no_future_leakage() -> None:
    market = market_frame()
    features = build_causal_features(market)
    prefix_features = build_causal_features(market.iloc[:280])
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert features["order_book_imbalance"].notna().all()
    assert features["taker_buy_sell_log_ratio"].notna().all()
    pd.testing.assert_frame_equal(
        features.iloc[:280].reset_index(drop=True),
        prefix_features.reset_index(drop=True),
    )


def test_microstructure_future_timestamp_is_rejected() -> None:
    market = market_frame()
    market.loc[12, "microstructure_timestamp"] += pd.Timedelta(hours=2)
    with pytest.raises(ValueError, match="outside_completed_candle"):
        build_causal_features(market)


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
    assert deflated_sharpe_ratio([0.02, -0.01, 0.03, 0.01], tested_strategies=10) is not None
    assert deflated_sharpe_ratio([0.01]) is None


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
        child.post_only_required
        and not child.exchange_enforced_post_only_confirmed
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
    risk_rejected = NobitexEnsemblePaperAdapter(
        FixedProbabilityModel(0.8)
    ).plan_entry(
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
    labels = (features["log_return_1"].to_numpy() > 0).astype(int)
    end_indices = [
        index + 24 if index + 24 < count else -1 for index in range(count)
    ]
    training_indices = [
        index for index, end in enumerate(end_indices) if end >= 0
    ]
    config = StackingConfig(
        fold_count=2,
        embargo_bars=24,
        minimum_train_samples=60,
        n_estimators=10,
        probability_threshold=0.5,
    )
    model = StackingEnsemble(config).fit(
        features,
        labels,
        end_indices,
        training_indices,
    )
    output = model.predict_proba(features.iloc[-10:])
    assert output.shape == (10,)
    assert np.all((output >= 0.0) & (output <= 1.0))
