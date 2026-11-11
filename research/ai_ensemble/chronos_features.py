"""Causal frozen Chronos-2 forecasts mapped as-of into hourly OHLCV features."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from research.ai_ensemble.features import REQUIRED_OHLCV_COLUMNS, build_causal_features

CHRONOS_MODEL_ID = "amazon/chronos-2"
CHRONOS_MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
CHRONOS_MODEL_WEIGHTS_SHA256 = (
    "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
)
CHRONOS_FORECASTING_VERSION = "2.3.2"
CHRONOS_CONTEXT_LENGTH = 512
CHRONOS_PREDICTION_LENGTH = 24
CHRONOS_REFRESH_CANDLES = 24
CHRONOS_BATCH_SIZE = 16
CHRONOS_QUANTILE_LEVELS = [0.1, 0.5, 0.9]
CHRONOS_FEATURE_COLUMN = "chronos_pred"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chronos_pipeline() -> tuple[Any, dict[str, object]]:
    try:
        import torch  # type: ignore[import-not-found]
        from chronos import BaseChronosPipeline  # type: ignore[import-not-found]
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("chronos_runtime_missing") from exc
    try:
        installed = version("chronos-forecasting")
    except PackageNotFoundError:
        installed = "unknown"
    if installed != CHRONOS_FORECASTING_VERSION:
        raise RuntimeError("chronos_forecasting_version_mismatch")
    torch.set_num_threads(1)
    weights = Path(
        hf_hub_download(
            repo_id=CHRONOS_MODEL_ID,
            filename="model.safetensors",
            revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
        )
    )
    weights_sha256 = _sha256_file(weights)
    if weights_sha256 != CHRONOS_MODEL_WEIGHTS_SHA256:
        raise RuntimeError("chronos_model_weight_sha256_mismatch")
    pipeline = BaseChronosPipeline.from_pretrained(
        CHRONOS_MODEL_ID,
        revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
        device_map="cpu",
    )
    return pipeline, {
        "model_id": CHRONOS_MODEL_ID,
        "model_revision": CHRONOS_MODEL_REVISION,
        "model_license": "Apache-2.0",
        "chronos_forecasting_version": installed,
        "torch_version": str(torch.__version__),
        "weight_file": "model.safetensors",
        "weight_sha256": weights_sha256,
        "device": "cpu",
        "context_length": CHRONOS_CONTEXT_LENGTH,
        "prediction_length": CHRONOS_PREDICTION_LENGTH,
        "forecast_refresh_candles": CHRONOS_REFRESH_CANDLES,
        "batch_size": CHRONOS_BATCH_SIZE,
        "quantile_levels": CHRONOS_QUANTILE_LEVELS,
        "forecast_type": "zero_shot",
    }


def _prepare_context(data: DataFrame) -> tuple[DataFrame, np.ndarray]:
    missing = sorted(set(REQUIRED_OHLCV_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"chronos_missing_columns:{','.join(missing)}")
    build_causal_features(data)
    frame = data.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy().reset_index(drop=True)
    timestamps_utc = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    if timestamps_utc.isna().any() or timestamps_utc.duplicated().any():
        raise ValueError("chronos_timestamp_invalid_or_duplicate")
    if not timestamps_utc.is_monotonic_increasing:
        raise ValueError("chronos_timestamps_not_increasing")
    if len(timestamps_utc) > 1 and not bool(
        (timestamps_utc.diff().iloc[1:] == pd.Timedelta(hours=1)).all()
    ):
        raise ValueError("chronos_requires_contiguous_hourly_ohlcv")
    # Chronos 2 normalizes timestamps through ndarray.view("int64").
    # Use UTC-naive datetime64 rather than timezone-aware object arrays.
    timestamps = timestamps_utc.dt.tz_localize(None)
    opened = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(frame["volume"], errors="coerce").to_numpy(dtype=float)
    if not all(np.isfinite(x).all() for x in (opened, high, low, close, volume)):
        raise ValueError("chronos_ohlcv_non_finite")
    if (close <= 0).any() or (volume < 0).any():
        raise ValueError("chronos_ohlcv_out_of_range")
    log_close = np.log(close)
    previous = np.roll(close, 1)
    previous[0] = close[0]
    prepared = DataFrame(
        {
            "timestamp": timestamps,
            "target": log_close,
            "log_return_1": np.diff(log_close, prepend=log_close[0]),
            "open_gap": opened / previous - 1.0,
            "range_fraction": (high - low) / close,
            "log_volume": np.log1p(volume),
        }
    )
    if not bool(
        np.isfinite(prepared.drop(columns="timestamp").to_numpy(dtype=float)).all()
    ):
        raise ValueError("chronos_covariates_non_finite")
    return prepared, log_close


def _build_context_batch(
    prepared: DataFrame, origins: Sequence[int], length: int
) -> DataFrame:
    parts = []
    for origin in origins:
        start = int(origin) - length + 1
        part = prepared.iloc[start : int(origin) + 1].copy()
        if start < 0 or len(part) != length:
            raise ValueError("chronos_context_warmup_incomplete")
        part.insert(0, "item_id", int(origin))
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    if result.isna().any().any():
        raise ValueError("chronos_context_contains_missing_values")
    return result


def build_chronos_features(
    ohlcv: DataFrame,
    *,
    pipeline: Any | None = None,
    context_length: int = CHRONOS_CONTEXT_LENGTH,
    prediction_length: int = CHRONOS_PREDICTION_LENGTH,
    refresh_candles: int = CHRONOS_REFRESH_CANDLES,
    batch_size: int = CHRONOS_BATCH_SIZE,
) -> DataFrame:
    """Create as-of hourly log-return forecasts using 512 past candles and 24h paths."""
    for name, value in (
        ("context_length", context_length),
        ("prediction_length", prediction_length),
        ("refresh_candles", refresh_candles),
        ("batch_size", batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"chronos_{name}_invalid")
    if prediction_length != refresh_candles:
        raise ValueError("chronos_prediction_refresh_horizon_mismatch")
    prepared, log_close = _prepare_context(ohlcv)
    if len(prepared) < context_length:
        raise ValueError("chronos_insufficient_history")
    active = pipeline
    if active is None:
        active, _ = load_chronos_pipeline()
    origins = list(range(context_length - 1, len(prepared), refresh_candles))
    values = np.full(len(prepared), np.nan, dtype=float)
    for offset in range(0, len(origins), batch_size):
        batch_origins = origins[offset : offset + batch_size]
        forecast = active.predict_df(
            _build_context_batch(prepared, batch_origins, context_length),
            prediction_length=prediction_length,
            quantile_levels=CHRONOS_QUANTILE_LEVELS,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            batch_size=min(batch_size, len(batch_origins)),
            context_length=context_length,
            cross_learning=False,
            validate_inputs=True,
            freq="h",
        )
        if not isinstance(forecast, DataFrame) or not {
            "item_id",
            "timestamp",
            "predictions",
        }.issubset(forecast.columns):
            raise ValueError("chronos_output_schema_invalid")
        for origin in batch_origins:
            path = forecast.loc[forecast["item_id"] == origin].sort_values("timestamp")
            if len(path) != prediction_length:
                raise ValueError("chronos_forecast_horizon_length_mismatch")
            actual = (
                pd.to_datetime(path["timestamp"], errors="coerce", utc=True)
                .dt.tz_localize(None)
                .tolist()
            )
            expected = pd.date_range(
                prepared["timestamp"].iloc[origin] + pd.Timedelta(hours=1),
                periods=prediction_length,
                freq="h",
            ).tolist()
            if actual != expected:
                raise ValueError("chronos_forecast_timestamp_mismatch")
            predicted = pd.to_numeric(path["predictions"], errors="coerce").to_numpy(
                dtype=float
            )
            if not bool(np.isfinite(predicted).all()):
                raise ValueError("chronos_forecast_non_finite")
            returns = np.diff(np.concatenate(([log_close[origin]], predicted)))
            count = min(prediction_length, len(prepared) - origin)
            values[origin : origin + count] = returns[:count]
    if not bool(np.isfinite(values[context_length - 1 :]).all()):
        raise ValueError("chronos_feature_coverage_incomplete")
    return DataFrame({"chronos_pred": values}, index=ohlcv.index)
