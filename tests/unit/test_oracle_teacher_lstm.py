from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.oracle_teacher_lstm import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    apply_partition_horizon_mask,
    build_causal_features,
    generate_oracle_labels,
)


def _frame(rows: int = 1200) -> pd.DataFrame:
    close = 100.0 + np.sin(np.arange(rows, dtype=np.float64) / 9.0)
    open_price = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "open": open_price,
            "high": np.maximum(open_price, close) + 0.2,
            "low": np.minimum(open_price, close) - 0.2,
            "close": close,
            "volume": 100.0 + (np.arange(rows) % 13),
        }
    )


def test_oracle_labels_first_barrier_and_timeout() -> None:
    frame = _frame(rows=40)
    frame.loc[1, ["open", "high", "low", "close"]] = [100.0, 103.5, 99.5, 101.0]
    frame.loc[4, ["open", "high", "low", "close"]] = [100.0, 100.5, 98.0, 99.0]
    frame.loc[7, ["open", "high", "low", "close"]] = [100.0, 102.9, 99.5, 101.0]
    frame.loc[8, ["open", "high", "low", "close"]] = [101.0, 102.8, 100.0, 102.0]

    labels = generate_oracle_labels(frame, horizon=2)

    assert labels.loc[0, "oracle_action"] == ACTION_BUY
    assert labels.loc[0, "barrier_outcome"] == "take_profit_first"
    assert labels.loc[3, "oracle_action"] == ACTION_SELL
    assert labels.loc[3, "barrier_outcome"] == "stop_loss_first"
    assert labels.loc[6, "oracle_action"] == ACTION_HOLD
    assert labels.loc[6, "barrier_outcome"] == "timeout_hold"
    assert labels["label_valid"].iloc[-2:].sum() == 0


def test_same_candle_double_touch_uses_conservative_stop_first_rule() -> None:
    frame = _frame(rows=40)
    frame.loc[1, ["open", "high", "low", "close"]] = [100.0, 104.0, 97.0, 100.0]

    labels = generate_oracle_labels(frame, horizon=2)

    assert labels.loc[0, "oracle_action"] == ACTION_SELL
    assert labels.loc[0, "barrier_outcome"] == ("same_candle_both_barriers_stop_first")


def test_partition_mask_removes_every_cross_boundary_target() -> None:
    row_count = 1200
    train_end = 840
    validation_end = 1020
    horizon = 24
    raw_valid = np.ones(row_count, dtype=np.bool_)

    valid = apply_partition_horizon_mask(
        raw_valid,
        row_count=row_count,
        train_end=train_end,
        validation_end=validation_end,
        horizon=horizon,
    )

    assert valid[train_end - horizon - 1]
    assert not valid[train_end - horizon]
    assert valid[validation_end - horizon - 1]
    assert not valid[validation_end - horizon]
    assert valid[row_count - horizon - 1]
    assert not valid[row_count - horizon]
    assert valid[0]


def test_causal_features_do_not_change_when_future_rows_are_modified() -> None:
    frame = _frame(rows=1200)
    original_features, feature_names = build_causal_features(frame)
    cutoff = 500
    changed = frame.copy()
    changed.loc[cutoff:, "open"] *= 1.2
    changed.loc[cutoff:, "high"] *= 1.2
    changed.loc[cutoff:, "low"] *= 1.2
    changed.loc[cutoff:, "close"] *= 1.2
    changed.loc[cutoff:, "volume"] *= 2.0
    changed_features, changed_names = build_causal_features(changed)

    assert feature_names == changed_names
    np.testing.assert_allclose(
        original_features[:cutoff],
        changed_features[:cutoff],
        rtol=0.0,
        atol=1e-7,
    )


def test_oracle_rejects_invalid_configuration_and_market_data() -> None:
    frame = _frame(rows=40)
    with pytest.raises(ValueError, match="oracle_horizon_out_of_range"):
        generate_oracle_labels(frame, horizon=0)
    with pytest.raises(ValueError, match="oracle_barrier_fraction_out_of_range"):
        generate_oracle_labels(frame, take_profit_fraction=0.5)
    broken = frame.copy()
    broken.loc[10, "high"] = np.nan
    with pytest.raises(ValueError, match="oracle_market_frame_non_finite"):
        generate_oracle_labels(broken)
