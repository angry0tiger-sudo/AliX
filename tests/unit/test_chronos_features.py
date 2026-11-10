from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from research.ai_ensemble.chronos_features import (
    CHRONOS_FEATURE_COLUMN,
    build_chronos_features,
)


class FakeChronosPipeline:
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
        assert pd.api.types.is_datetime64_any_dtype(context[timestamp_column].dtype)
        output: list[dict[str, Any]] = []
        for item_id, group in context.groupby(id_column, sort=False):
            value = float(group[target].iloc[-1])
            timestamp = pd.Timestamp(group[timestamp_column].iloc[-1])
            for step in range(1, prediction_length + 1):
                output.append(
                    {
                        id_column: item_id,
                        timestamp_column: timestamp + pd.Timedelta(hours=step),
                        "predictions": value + step * 0.0001,
                    }
                )
        return DataFrame.from_records(output)


def market_frame(rows: int = 180) -> DataFrame:
    rng = np.random.default_rng(88)
    timestamps = pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="h")
    close = np.empty(rows)
    opened = np.empty(rows)
    close[0] = opened[0] = 100.0
    for index in range(1, rows):
        opened[index] = close[index - 1]
        close[index] = opened[index] * np.exp(0.0001 + rng.normal(0.0, 0.002))
    return DataFrame(
        {
            "date": timestamps,
            "open": opened,
            "high": np.maximum(opened, close) * 1.003,
            "low": np.minimum(opened, close) * 0.997,
            "close": close,
            "volume": rng.lognormal(4.0, 0.25, rows),
        }
    )


def test_chronos_feature_is_causal_and_asof() -> None:
    source = market_frame()
    first = build_chronos_features(
        source,
        pipeline=FakeChronosPipeline(),
        context_length=32,
        prediction_length=24,
        refresh_candles=24,
        batch_size=3,
    )
    mutated = source.copy()
    mutated.loc[100:, ["open", "high", "low", "close", "volume"]] *= 2.0
    second = build_chronos_features(
        mutated,
        pipeline=FakeChronosPipeline(),
        context_length=32,
        prediction_length=24,
        refresh_candles=24,
        batch_size=3,
    )
    assert first[CHRONOS_FEATURE_COLUMN].iloc[:31].isna().all()
    assert first[CHRONOS_FEATURE_COLUMN].iloc[31:].notna().all()
    np.testing.assert_allclose(
        first[CHRONOS_FEATURE_COLUMN].iloc[:100].to_numpy(dtype=float),
        second[CHRONOS_FEATURE_COLUMN].iloc[:100].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
    )


def test_chronos_rejects_noncontiguous_hourly_input() -> None:
    market = market_frame().drop(index=50).reset_index(drop=True)
    with pytest.raises(ValueError, match="contiguous"):
        build_chronos_features(
            market,
            pipeline=FakeChronosPipeline(),
            context_length=32,
            prediction_length=24,
            refresh_candles=24,
            batch_size=2,
        )


def test_chronos_rejects_incomplete_forecast() -> None:
    class InvalidPipeline:
        def predict_df(self, *_args: Any, **_kwargs: Any) -> DataFrame:
            return DataFrame(
                {
                    "item_id": [31],
                    "timestamp": [pd.Timestamp("2024-01-01")],
                    "predictions": [np.nan],
                }
            )

    with pytest.raises(ValueError, match="horizon_length_mismatch"):
        build_chronos_features(
            market_frame(100),
            pipeline=InvalidPipeline(),
            context_length=32,
            prediction_length=24,
            refresh_candles=24,
            batch_size=2,
        )
