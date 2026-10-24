from __future__ import annotations

import pandas as pd

from scripts.run_stage_b_walk_forward import (
    FEATURE_COLUMNS,
    add_market_features,
    build_fixed_window_folds,
    economic_gate,
    segment_bounds,
    select_segment_features,
)


def test_segment_bounds_split_at_gap_without_inserting_rows() -> None:
    timestamps = [1800, 5400, 9000, 23400, 27000]
    ranges, gaps = segment_bounds(timestamps)
    assert ranges == [(0, 3), (3, 5)]
    assert gaps == [
        {
            "left_timestamp": 9000,
            "right_timestamp": 23400,
            "missing_hours": 3,
        }
    ]
    assert sum(stop - start for start, stop in ranges) == len(timestamps)


def test_segment_bounds_reject_duplicate_or_subhour_ordering() -> None:
    for timestamps in ([1800, 5400, 5400], [1800, 5400, 7200]):
        try:
            segment_bounds(timestamps)
        except RuntimeError:
            continue
        raise AssertionError("invalid timestamps must fail closed")


def test_fixed_windows_expand_training_and_purge_forward_label_horizon() -> None:
    folds = build_fixed_window_folds(
        stop_index=11500,
        initial_train_rows=4320,
        horizon=12,
        window_rows=2160,
        prefix="B-DIAG",
    )
    assert len(folds) == 4
    assert folds[0].fold_id == "B-DIAG-01"
    assert folds[0].train_stop == 4320
    assert folds[0].test_start == 4332
    assert folds[0].test_stop == 6492
    assert folds[0].train_stop + 12 == folds[0].test_start
    assert folds[-1].test_stop == 11500
    assert folds[-1].test_stop - folds[-1].test_start < 2160
    for previous, current in zip(folds, folds[1:]):
        assert previous.test_stop == current.test_start
        assert current.train_stop > previous.train_stop


def test_economic_gate_uses_strict_win_rate_and_drawdown_inequalities() -> None:
    passes = economic_gate(100, 0.61, 0.001, 0.149)
    assert passes["status"] == "THRESHOLDS_MET_RESEARCH_ONLY"

    exact_win_rate = economic_gate(100, 0.60, 0.001, 0.10)
    assert exact_win_rate["status"] == "NO_GO_THRESHOLDS_NOT_MET"
    assert exact_win_rate["win_rate_pass"] is False

    exact_drawdown = economic_gate(100, 0.61, 0.001, 0.15)
    assert exact_drawdown["status"] == "NO_GO_THRESHOLDS_NOT_MET"
    assert exact_drawdown["drawdown_pass"] is False


def test_economic_gate_never_passes_with_fewer_than_100_oos_trades() -> None:
    result = economic_gate(99, 1.0, 1.0, 0.0)
    assert result["status"] == "INSUFFICIENT_SAMPLE"
    assert result["trade_count_pass"] is False


def test_unavailable_feature_is_excluded_without_imputation() -> None:
    frame = pd.DataFrame(
        {
            "available": [float("nan"), 1.0, 2.0, 3.0, 4.0],
            "undefined_inside_segment": [
                float("nan"),
                1.0,
                float("nan"),
                3.0,
                4.0,
            ],
        }
    )
    selected, excluded = select_segment_features(
        frame, ("available", "undefined_inside_segment")
    )
    assert selected == ["available"]
    assert excluded == [
        {
            "feature": "undefined_inside_segment",
            "reason": "undefined_on_otherwise_usable_rows",
        }
    ]
    assert frame["undefined_inside_segment"].isna().sum() == 2


def test_feature_selection_ignores_staggered_leading_warmup_but_rejects_internal_gap() -> None:
    frame = pd.DataFrame(
        {
            "return_1": [float("nan"), 0.1, 0.2, 0.3, 0.4, 0.5],
            "return_3": [
                float("nan"),
                float("nan"),
                float("nan"),
                0.3,
                0.4,
                0.5,
            ],
            "atr_proxy_14": [
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                0.4,
                0.5,
            ],
            "volume_change": [float("nan"), 0.1, 0.2, 0.3, float("nan"), 0.5],
        }
    )

    selected, excluded = select_segment_features(
        frame, ("return_1", "return_3", "atr_proxy_14", "volume_change")
    )

    assert selected == ["return_1", "return_3", "atr_proxy_14"]
    assert excluded == [
        {
            "feature": "volume_change",
            "reason": "undefined_on_otherwise_usable_rows",
        }
    ]
    assert frame.loc[:3, "atr_proxy_14"].isna().all()


def test_market_feature_schema_survives_natural_rolling_warmup() -> None:
    candle_count = 40
    close = pd.Series([100.0 + index for index in range(candle_count)])
    frame = pd.DataFrame(
        {
            "date": pd.date_range(
                "2020-01-01", periods=candle_count, freq="h", tz="UTC"
            ),
            "open": close - 0.25,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [float(index + 1) for index in range(candle_count)],
        }
    )

    engineered = add_market_features(frame)
    selected, excluded = select_segment_features(engineered)

    assert selected == list(FEATURE_COLUMNS)
    assert excluded == []
    assert engineered.loc[14, list(FEATURE_COLUMNS)].notna().all()
