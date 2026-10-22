from __future__ import annotations

import pandas as pd

from services.data_quality.gate import QualityReasonCode, QualityStatus
from user_data.strategies.AliXStrategy import AliXStrategy


def market_frame(hours: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [f"2026-10-22T{hour:02d}:00:00Z" for hour in hours], utc=True
            ),
            "open": [10.0] * len(hours),
            "high": [12.0] * len(hours),
            "low": [9.0] * len(hours),
            "close": [11.0] * len(hours),
            "volume": [5.0] * len(hours),
        }
    )


def test_strategy_runs_valid_ohlcv_gate_before_feature_stage() -> None:
    strategy = AliXStrategy()

    checked = strategy.populate_indicators(
        market_frame([0, 1]), {"pair": "BTC/USDT"}
    )

    assert checked["alix_data_quality_valid"].all()
    assert checked.attrs["alix_data_quality"]["status"] == QualityStatus.VALID.value
    assert checked.attrs["alix_data_quality"]["fingerprint"]


def test_strategy_marks_gap_window_invalid_and_keeps_entries_closed() -> None:
    strategy = AliXStrategy()
    checked = strategy.populate_indicators(
        market_frame([0, 2]), {"pair": "BTC/USDT"}
    )
    signaled = strategy.populate_entry_trend(checked.copy(), {"pair": "BTC/USDT"})

    assert not checked["alix_data_quality_valid"].any()
    assert QualityReasonCode.DATA_GAP.value in checked.attrs["alix_data_quality"][
        "reason_codes"
    ]
    assert signaled["enter_long"].eq(0).all()


def test_strategy_fails_closed_when_required_timestamp_is_missing() -> None:
    strategy = AliXStrategy()
    missing_date = market_frame([0, 1]).drop(columns=["date"])

    checked = strategy.populate_indicators(
        missing_date, {"pair": "BTC/USDT"}
    )

    assert not checked["alix_data_quality_valid"].any()
    assert checked.attrs["alix_data_quality"]["status"] == QualityStatus.REJECTED.value
