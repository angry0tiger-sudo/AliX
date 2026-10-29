from __future__ import annotations

from inspect import signature

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from services.ml_core.adaptive_targets import (
    ATR_PERIOD,
    EXIT_REASONS,
    K_LOWER_DEFAULT,
    K_UPPER_DEFAULT,
    LABEL_HORIZON_CANDLES,
    apply_adaptive_triple_barrier,
    calculate_atr,
)


def market_frame(rows: int = 45) -> DataFrame:
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="h")
    close = np.full(rows, 100.0)
    return DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(rows, 10.0),
        }
    )


def constant_atr(frame: DataFrame, value: float) -> pd.Series:
    return pd.Series(np.full(len(frame), value, dtype=float))


def test_atr_uses_true_range_and_full_period_mean_on_known_inputs() -> None:
    candles = market_frame(15)
    candles.loc[14, "high"] = 103.5
    atr = calculate_atr(candles, period=14)

    assert ATR_PERIOD == 14
    assert atr.iloc[:13].isna().all()
    assert atr.iloc[13] == pytest.approx(1.0)
    assert atr.iloc[14] == pytest.approx((13.0 * 1.0 + 4.0) / 14.0)


def test_default_target_contract_is_tp_two_atr_sl_one_atr_horizon_24() -> None:
    candles = market_frame(45)
    atr = constant_atr(candles, 1.0)
    labels = apply_adaptive_triple_barrier(candles, atr_values=atr)

    assert K_UPPER_DEFAULT == 2.0
    assert K_LOWER_DEFAULT == 1.0
    assert LABEL_HORIZON_CANDLES == 24
    row = labels.iloc[10]
    assert row["take_profit_price"] == pytest.approx(102.0)
    assert row["stop_loss_price"] == pytest.approx(99.0)
    assert row["label"] == "HOLD"
    assert row["exit_reason"] == "HORIZON_EXPIRY"
    assert row["label_end_idx"] == 34


def test_labeler_signature_and_label_end_idx_contract() -> None:
    parameters = signature(apply_adaptive_triple_barrier).parameters
    assert "k_upper" in parameters
    assert "k_lower" in parameters

    candles = market_frame(40)
    labels = apply_adaptive_triple_barrier(
        candles, atr_values=constant_atr(candles, 1.0)
    )
    assert "label_end_idx" in labels.columns
    assert "label_end_index" not in labels.columns


def test_same_candle_collision_resolves_to_sell_stop_loss() -> None:
    candles = market_frame(45)
    candles.loc[15, "high"] = 103.0
    candles.loc[15, "low"] = 98.0
    labels = apply_adaptive_triple_barrier(
        candles, atr_values=constant_atr(candles, 1.0)
    )

    row = labels.iloc[14]
    assert row["take_profit_price"] == pytest.approx(102.0)
    assert row["stop_loss_price"] == pytest.approx(99.0)
    assert row["label"] == "SELL"
    assert row["exit_reason"] == "COLLISION"
    assert row["label_end_idx"] == 15


def test_first_take_profit_hit_is_buy_and_stop_loss_hit_is_sell() -> None:
    candles = market_frame(12)
    candles.loc[1, "high"] = 103.0
    candles.loc[3, "low"] = 98.0
    labels = apply_adaptive_triple_barrier(
        candles,
        atr_values=constant_atr(candles, 1.0),
        horizon=3,
        atr_period=1,
    )

    assert labels.iloc[0]["label"] == "BUY"
    assert labels.iloc[0]["exit_reason"] == "TP_HIT"
    assert labels.iloc[2]["label"] == "SELL"
    assert labels.iloc[2]["exit_reason"] == "SL_HIT"


def test_horizon_expiry_is_hold_and_tail_without_full_horizon_is_unlabeled() -> None:
    candles = market_frame(40)
    labels = apply_adaptive_triple_barrier(
        candles, atr_values=constant_atr(candles, 1.0), horizon=24
    )

    assert labels.iloc[14]["label"] == "HOLD"
    assert labels.iloc[14]["exit_reason"] == "horizon_expiry"
    assert labels.iloc[14]["label_end_idx"] == 38
    assert labels.iloc[-1]["label"] is None
    assert labels.iloc[-1]["exit_reason"] is None
    assert labels.iloc[-1]["unlabeled_reason"] == "incomplete_horizon"
    assert set(labels["exit_reason"].dropna().unique()) <= set(EXIT_REASONS)


def test_label_distribution_fixture_contains_buy_sell_and_hold() -> None:
    candles = market_frame(12)
    candles.loc[1, "high"] = 103.0
    candles.loc[3, "low"] = 98.0
    labels = apply_adaptive_triple_barrier(
        candles,
        atr_values=constant_atr(candles, 1.0),
        horizon=3,
        atr_period=1,
    )
    observed = set(labels["label"].dropna().tolist())

    assert observed == {"BUY", "SELL", "HOLD"}
    assert observed != {"HOLD"}
    assert observed != {"BUY"}


def test_zero_nan_and_extreme_atr_fail_closed() -> None:
    candles = market_frame(40)
    with pytest.raises(ValueError, match="atr_must_be_positive"):
        apply_adaptive_triple_barrier(candles, atr_values=constant_atr(candles, 0.0))

    invalid_atr = constant_atr(candles, 1.0)
    invalid_atr.iloc[14] = float("nan")
    with pytest.raises(ValueError, match="atr_must_be_finite"):
        apply_adaptive_triple_barrier(candles, atr_values=invalid_atr)

    # Very wide but finite barriers remain valid and result in HOLD on a quiet path.
    extreme = apply_adaptive_triple_barrier(
        candles, atr_values=constant_atr(candles, 49.0)
    )
    assert extreme.iloc[14]["label"] == "HOLD"

    with pytest.raises(ValueError, match="atr_barrier_price_not_finite"):
        apply_adaptive_triple_barrier(candles, atr_values=constant_atr(candles, 1e308))


def test_computed_atr_warmup_is_unlabeled_and_flat_market_rejects_zero_atr() -> None:
    candles = market_frame(40)
    labels = apply_adaptive_triple_barrier(candles)

    assert labels.iloc[0]["label"] is None
    assert labels.iloc[0]["exit_reason"] is None
    assert labels.iloc[0]["unlabeled_reason"] == "insufficient_atr_history"

    flat = candles.copy()
    flat["high"] = flat["close"]
    flat["low"] = flat["close"]
    with pytest.raises(ValueError, match="atr_must_be_positive"):
        apply_adaptive_triple_barrier(flat)


def test_invalid_configuration_and_malformed_ohlcv_fail_closed() -> None:
    candles = market_frame(40)
    with pytest.raises(ValueError, match="horizon_must_be_positive_integer"):
        apply_adaptive_triple_barrier(candles, horizon=0)
    with pytest.raises(ValueError, match="atr_period_must_be_positive_integer"):
        calculate_atr(candles, period=0)
    with pytest.raises(ValueError, match="barrier_multipliers"):
        apply_adaptive_triple_barrier(candles, k_upper=float("nan"))
    with pytest.raises(ValueError, match="atr_length_mismatch"):
        apply_adaptive_triple_barrier(candles, atr_values=pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="timezone_required"):
        naive = candles.copy()
        naive["date"] = naive["date"].dt.tz_localize(None)
        calculate_atr(naive)
    with pytest.raises(ValueError, match="ohlcv_non_finite:close"):
        invalid = candles.copy()
        invalid.loc[2, "close"] = float("nan")
        calculate_atr(invalid)
