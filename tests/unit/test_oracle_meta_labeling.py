from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.oracle_teacher_lstm import ACTION_BUY, ACTION_HOLD, ACTION_SELL

from scripts.oracle_meta_labeling import (
    _binary_metrics,
    _chronological_boundaries,
    _gap_exit,
    _intrabar_exit,
    _meta_feature_matrix,
    _partition_masks,
    _profitability_targets,
    _simulate_long_flat,
)


def _frame(rows: int = 300) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 100.2),
            "low": np.full(rows, 99.8),
            "close": np.full(rows, 100.0),
            "volume": np.full(rows, 1000.0),
        }
    )


def test_chronological_partitions_are_ordered_and_disjoint() -> None:
    boundaries = _chronological_boundaries(1000)
    assert boundaries["base_train_end"] == 500
    assert boundaries["base_validation_end"] == 550
    assert boundaries["meta_train_end"] == 700
    assert boundaries["validation_end"] == 850
    assert boundaries["holdout_end"] == 1000
    with pytest.raises(ValueError, match="meta_labeling_temporal_partition_too_short"):
        _chronological_boundaries(100)


def test_partition_masks_purge_forward_labels_at_each_boundary() -> None:
    boundaries = {
        "base_train_end": 100,
        "base_validation_end": 160,
        "meta_train_end": 230,
        "validation_end": 300,
        "holdout_end": 400,
    }
    rows = np.arange(400, dtype=np.int64)
    masks = _partition_masks(rows, np.ones(400, dtype=np.bool_), boundaries, horizon=24)
    assert masks["base_train"][75] and not masks["base_train"][76]
    assert masks["base_validation"][135] and not masks["base_validation"][136]
    assert masks["meta_train"][205] and not masks["meta_train"][206]
    assert masks["validation"][275] and not masks["validation"][276]
    assert masks["holdout"][375] and not masks["holdout"][376]


def test_meta_feature_matrix_has_probabilities_atr_and_ma_distances() -> None:
    probabilities = np.asarray([[0.2, 0.6, 0.2]], dtype=np.float32)
    names = [
        "return_1",
        "atr_fraction_14",
        "sma_distance_6",
        "sma_distance_24",
        "sma_distance_72",
        "sma_distance_168",
    ]
    features = np.asarray([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]], dtype=np.float32)
    matrix, meta_names = _meta_feature_matrix(probabilities, features, names)
    assert meta_names[0:3] == [
        "lstm_probability_buy",
        "lstm_probability_sell",
        "lstm_probability_hold",
    ]
    np.testing.assert_allclose(matrix[0], [0.6, 0.2, 0.2, 0.02, 0.03, 0.04, 0.05, 0.06])


def test_binary_metrics_calculate_macro_f1() -> None:
    result = _binary_metrics(
        np.asarray([1, 1, 0, 0], dtype=np.int64),
        np.asarray([1, 0, 0, 1], dtype=np.int64),
    )
    assert result["sample_count"] == 4
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_negative"] == 1
    assert result["macro_f1"] == pytest.approx(0.5)


def test_profitability_target_accounts_for_barriers_and_costs() -> None:
    frame = _frame(80)
    frame.loc[20, "close"] = 104.0
    frame.loc[30, "close"] = 90.0
    labels = pd.DataFrame(
        {
            "oracle_action": np.full(80, ACTION_HOLD, dtype=np.int64),
            "entry_price": np.full(80, 100.0),
            "label_valid": np.arange(80, dtype=np.int64) + 24 < 80,
        }
    )
    labels.loc[1, "oracle_action"] = ACTION_BUY
    labels.loc[2, "oracle_action"] = ACTION_SELL
    targets, returns = _profitability_targets(frame, labels, horizon=24)
    assert targets[1] == 1
    assert targets[2] == 0
    assert returns[1] == pytest.approx(0.03 - 0.009)
    assert returns[2] == pytest.approx(-0.015 - 0.009)
    assert np.isnan(returns[-1])


def test_barrier_tie_is_stop_first_and_gaps_are_conservative() -> None:
    position = {
        "entry_index": 10,
        "entry_price": 100.0,
        "entry_equity": 1.0,
        "quantity": 0.009,
    }
    assert _gap_exit(98.0, position) == ("GAP_STOP", 98.0)
    assert _gap_exit(104.0, position) == ("GAP_TAKE_PROFIT", 104.0)
    assert _gap_exit(100.0, position) is None
    assert _intrabar_exit(104.0, 97.0, position) == ("SAME_CANDLE_STOP_FIRST", 98.5)
    assert _intrabar_exit(104.0, 99.0, position) == ("TAKE_PROFIT", 103.0)
    assert _intrabar_exit(102.0, 99.0, position) is None


def test_meta_filter_does_not_enter_short_or_approve_low_probability_buy() -> None:
    frame = _frame(80)
    actions = np.full(80, ACTION_HOLD, dtype=np.int64)
    probabilities = np.zeros(80, dtype=np.float64)
    actions[1] = ACTION_BUY
    probabilities[1] = 0.2
    actions[2] = ACTION_SELL
    result = _simulate_long_flat(
        frame, actions, probabilities, threshold=0.5, start=0, end=40, horizon=24
    )
    assert result["closed_trade_count"] == 0
    assert result["net_return_fraction"] == pytest.approx(0.0)


def test_meta_filter_executes_approved_buy_and_respects_take_profit() -> None:
    frame = _frame(100)
    actions = np.full(100, ACTION_HOLD, dtype=np.int64)
    probabilities = np.zeros(100, dtype=np.float64)
    actions[1] = ACTION_BUY
    probabilities[1] = 0.9
    frame.loc[2, ["high", "low", "close"]] = [104.0, 99.0, 103.0]
    result = _simulate_long_flat(
        frame, actions, probabilities, threshold=0.5, start=0, end=50, horizon=24
    )
    assert result["closed_trade_count"] == 1
    assert result["trades"][0]["exit_reason"] == "TAKE_PROFIT"
    assert result["net_return_fraction"] > 0
    assert result["fee_bps_per_side"] == 25.0


def test_meta_economic_simulation_rejects_bad_input() -> None:
    frame = _frame(40)
    with pytest.raises(
        ValueError, match="meta_economic_simulation_configuration_invalid"
    ):
        _simulate_long_flat(
            frame,
            np.zeros(30, dtype=np.int64),
            np.zeros(40),
            threshold=0.5,
            start=0,
            end=20,
        )
