from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from services.rl_research.rl_data_curriculum import select_high_volatility_window


def _market_data_from_returns(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    close = np.empty(len(returns) + 1, dtype=np.float64)
    close[0] = 100.0
    close[1:] = close[0] * np.exp(np.cumsum(returns))
    timestamps = 1_700_000_000_000 + np.arange(len(close)) * 60_000
    return np.column_stack(
        (timestamps, close, close, close, close, np.ones(len(close)))
    ).astype(np.float64)


def test_selects_a_directional_window_from_the_high_volatility_quartile() -> None:
    returns = np.full(399, 0.00001, dtype=np.float64)
    returns[150:250] = np.tile(
        np.asarray([0.003, -0.0005, 0.002, -0.0004], dtype=np.float64), 25
    )
    data = _market_data_from_returns(returns)

    selected, metadata = select_high_volatility_window(
        data,
        window_candles=60,
        candidate_stride_candles=10,
        minimum_volatility_quantile=0.75,
    )

    start = int(metadata["selected_start_index"])
    end = int(metadata["selected_end_index_exclusive"])
    assert len(selected) == 60
    assert start < 250 and end > 150
    assert metadata["selection_split"] == "training_only"
    assert metadata["candidate_window_count"] > 1
    assert (
        metadata["selected_realized_volatility_per_candle"]
        >= (metadata["volatility_threshold_per_candle"])
    )
    assert metadata["selected_net_return_pct"] > 0.0
    assert int(selected[0, 0]) == int(metadata["selected_start_timestamp_ms"])
    assert int(selected[-1, 0]) == int(metadata["selected_end_timestamp_ms"])


@pytest.mark.parametrize(
    ("window_candles", "stride", "message"),
    [
        (1, 10, "regime_window_candles_out_of_range"),
        (500, 10, "regime_window_candles_out_of_range"),
        (20, 0, "candidate_stride_candles_must_be_positive"),
    ],
)
def test_rejects_invalid_window_configuration(
    window_candles: int, stride: int, message: str
) -> None:
    data = _market_data_from_returns(np.full(99, 0.001, dtype=np.float64))
    with pytest.raises(ValueError, match=message):
        select_high_volatility_window(
            data, window_candles=window_candles, candidate_stride_candles=stride
        )


def test_rejects_non_increasing_timestamps() -> None:
    data = _market_data_from_returns(np.full(19, 0.001, dtype=np.float64))
    data[8, 0] = data[7, 0]
    with pytest.raises(ValueError, match="market_data_timestamps_must_increase"):
        select_high_volatility_window(data, window_candles=10)
