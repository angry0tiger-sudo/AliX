"""Train a risk-only PPO allocator from an immutable five-year OHLCV artifact.

This script never downloads market data or contacts an exchange. It requires
BTCUSDT/ETHUSDT CSVs plus producer manifests and reserves the final 15% holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Series

from scripts.run_ml_core_v4ap_diagnostic_sweep import (
    generate_donchian_breakout_signal,
)
from services.ml_core.adaptive_targets import calculate_atr
from services.rl_research.risk_allocation_environment import (
    ACTION_FRACTIONS,
    ROUND_TRIP_COST_BPS,
    RiskAllocationEnv,
)

REQUESTED_DAYS = 1825
EXPECTED_MINUTES = REQUESTED_DAYS * 24 * 60
EXPECTED_SCHEMA = "alix.binance.public-ohlcv.v1"
EXPECTED_PROVIDER = "Binance Spot public klines API"
ASSETS = ("BTCUSDT", "ETHUSDT")
TIMEFRAME_MINUTES = {"15m": 15, "30m": 30, "1h": 60, "4h": 240}
FIXED_BASELINE_EXPOSURE_FRACTION = 1.0
CSV_COLUMNS = ("date", "open", "high", "low", "close", "volume")
KNOWN_GAP_START = "2023-03-24T12:40:00Z"
KNOWN_GAP_END = "2023-03-24T13:59:00Z"
KNOWN_GAP_PREVIOUS = "2023-03-24T12:39:00Z"
KNOWN_GAP_NEXT = "2023-03-24T14:00:00Z"
Float64Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AssetDataset:
    symbol: str
    candles: DataFrame
    manifest: dict[str, Any]
    csv_sha256: str


@dataclass(frozen=True, slots=True)
class AssetFeatures:
    symbol: str
    train_segments: tuple[DataFrame, ...]
    validation_segments: tuple[DataFrame, ...]
    manifest: dict[str, Any]


def _validate_commit_sha(value: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("rl_commit_sha_invalid")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _utc_timestamp(value: object, field_name: str) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"rl_dataset_timestamp_invalid:{field_name}")
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"rl_dataset_timestamp_invalid:{field_name}")
    return pd.Timestamp(timestamp)


def _validate_manifest(
    manifest: dict[str, Any], *, symbol: str, csv_digest: str
) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    start = _utc_timestamp(manifest.get("start_utc"), "start_utc")
    end = _utc_timestamp(manifest.get("end_utc"), "end_utc")
    missing = manifest.get("missing_minute_intervals")
    if isinstance(missing, bool) or not isinstance(missing, int):
        raise ValueError("rl_dataset_missing_minute_count_invalid")
    expected_gap = [
        {
            "start_utc": KNOWN_GAP_START,
            "end_utc": KNOWN_GAP_END,
            "missing_minute_intervals": 80,
        }
    ]
    expected_gap_ranges = [] if missing == 0 else expected_gap
    conditions = (
        manifest.get("schema_version") == EXPECTED_SCHEMA,
        manifest.get("provider") == EXPECTED_PROVIDER,
        manifest.get("symbol") == symbol,
        manifest.get("pair") == f"{symbol[:-4]}/USDT",
        manifest.get("interval") == "1m",
        manifest.get("requested_days") == REQUESTED_DAYS,
        end - start == pd.Timedelta(days=REQUESTED_DAYS) - pd.Timedelta(minutes=1),
        manifest.get("row_count") == EXPECTED_MINUTES - missing,
        missing in (0, 80),
        manifest.get("missing_gap_ranges") == expected_gap_ranges,
        manifest.get("csv_sha256") == csv_digest,
        isinstance(manifest.get("commit_sha"), str)
        and len(manifest["commit_sha"]) == 40,
        manifest.get("public_api_only") is True,
        manifest.get("credentials_used") is False,
        manifest.get("private_api_calls") == 0,
        manifest.get("exchange_calls") == 0,
        manifest.get("real_order_attempts") == 0,
        manifest.get("Champion") is None,
        manifest.get("order_permission") is False,
        manifest.get("trading_authorization") == "NO-GO",
    )
    if not all(conditions):
        raise ValueError(f"rl_dataset_manifest_contract_invalid:{symbol}")
    return start, end, missing


def _validate_candle_timestamp_sequence(
    data: DataFrame, *, symbol: str, expected_rows: int
) -> None:
    dates = data["date"]
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"rl_dataset_timestamps_invalid:{symbol}")
    if not dates.is_monotonic_increasing:
        raise ValueError(f"rl_dataset_timestamps_not_monotonic:{symbol}")
    if len(data) != expected_rows or data.empty:
        raise ValueError(f"rl_dataset_row_count_invalid:{symbol}")


def _validate_candle_window(
    data: DataFrame, *, symbol: str, start: pd.Timestamp, end: pd.Timestamp
) -> None:
    if (
        pd.Timestamp(data["date"].iloc[0]) != start
        or pd.Timestamp(data["date"].iloc[-1]) != end
    ):
        raise ValueError(f"rl_dataset_window_bounds_invalid:{symbol}")


def _validate_known_gap_location(
    data: DataFrame, *, symbol: str, irregular: Series
) -> None:
    if len(irregular) != 1 or irregular.iloc[0] != pd.Timedelta(minutes=81):
        raise ValueError(f"rl_dataset_gap_geometry_invalid:{symbol}")
    gap_index = int(irregular.index[0])
    previous = pd.Timestamp(data["date"].iloc[gap_index - 1])
    following = pd.Timestamp(data["date"].iloc[gap_index])
    if previous != pd.Timestamp(KNOWN_GAP_PREVIOUS) or following != pd.Timestamp(
        KNOWN_GAP_NEXT
    ):
        raise ValueError(f"rl_dataset_gap_location_invalid:{symbol}")


def _validate_candle_gap_geometry(
    data: DataFrame, *, symbol: str, missing_minutes: int
) -> None:
    differences = data["date"].diff().iloc[1:]
    irregular = differences[differences != pd.Timedelta(minutes=1)]
    if missing_minutes == 0:
        if not irregular.empty:
            raise ValueError(f"rl_dataset_unmanifested_gap:{symbol}")
        return
    if missing_minutes != 80:
        raise ValueError(f"rl_dataset_gap_policy_invalid:{symbol}")
    _validate_known_gap_location(data, symbol=symbol, irregular=irregular)


def _validate_ohlcv_geometry(data: DataFrame, *, symbol: str) -> None:
    opened = data["open"].to_numpy(dtype=np.float64)
    high = data["high"].to_numpy(dtype=np.float64)
    low = data["low"].to_numpy(dtype=np.float64)
    close = data["close"].to_numpy(dtype=np.float64)
    volume = data["volume"].to_numpy(dtype=np.float64)
    if np.any(np.minimum.reduce((opened, high, low, close)) <= 0.0):
        raise ValueError(f"rl_dataset_ohlcv_geometry_invalid:{symbol}")
    if np.any(volume < 0.0):
        raise ValueError(f"rl_dataset_ohlcv_geometry_invalid:{symbol}")
    if np.any(high < np.maximum(opened, close)):
        raise ValueError(f"rl_dataset_ohlcv_geometry_invalid:{symbol}")
    if np.any(low > np.minimum(opened, close)):
        raise ValueError(f"rl_dataset_ohlcv_geometry_invalid:{symbol}")


def _validate_candle_numeric_values(data: DataFrame, *, symbol: str) -> None:
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    values = data.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError(f"rl_dataset_numeric_values_invalid:{symbol}")
    _validate_ohlcv_geometry(data, symbol=symbol)


def _validate_candles(
    candles: DataFrame,
    *,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    missing_minutes: int,
    expected_rows: int,
) -> DataFrame:
    if tuple(candles.columns) != CSV_COLUMNS:
        raise ValueError(f"rl_dataset_csv_schema_invalid:{symbol}")
    data = candles.copy()
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce")
    _validate_candle_timestamp_sequence(
        data, symbol=symbol, expected_rows=expected_rows
    )
    _validate_candle_window(data, symbol=symbol, start=start, end=end)
    _validate_candle_gap_geometry(data, symbol=symbol, missing_minutes=missing_minutes)
    _validate_candle_numeric_values(data, symbol=symbol)
    return data


def load_asset_dataset(input_dir: Path, symbol: str) -> AssetDataset:
    if symbol not in ASSETS:
        raise ValueError("rl_dataset_symbol_not_allowed")
    stem = f"binance-{symbol.lower()}-1m"
    csv_path = input_dir / f"{stem}.csv"
    manifest_path = input_dir / f"{stem}.manifest.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"rl_dataset_artifact_missing:{symbol}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"rl_dataset_manifest_invalid:{symbol}")
    digest = _sha256(csv_path)
    start, end, missing = _validate_manifest(manifest, symbol=symbol, csv_digest=digest)
    candles = pd.read_csv(csv_path)
    data = _validate_candles(
        candles,
        symbol=symbol,
        start=start,
        end=end,
        missing_minutes=missing,
        expected_rows=int(manifest["row_count"]),
    )
    return AssetDataset(symbol, data, manifest, digest)


def resample_complete_bars(candles: DataFrame, timeframe: str) -> DataFrame:
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError("rl_entry_timeframe_not_allowed")
    minutes = TIMEFRAME_MINUTES[timeframe]
    indexed = candles.set_index("date")
    grouped = indexed.resample(
        f"{minutes}min", origin="epoch", label="left", closed="left"
    )
    bars = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    counts = grouped["close"].count()
    complete = bars.loc[counts == minutes].dropna().reset_index()
    if len(complete) < 256:
        raise ValueError(f"rl_resampled_dataset_too_short:{timeframe}")
    return complete.loc[:, list(CSV_COLUMNS)]


def _contiguous_segments(candles: DataFrame, interval: pd.Timedelta) -> list[DataFrame]:
    if candles.empty:
        return []
    deltas = candles["date"].diff().iloc[1:]
    boundaries = [
        int(index) for index in np.flatnonzero(deltas.to_numpy() != interval) + 1
    ]
    starts = [0, *boundaries]
    ends = [*boundaries, len(candles)]
    return [
        candles.iloc[start:end].reset_index(drop=True)
        for start, end in zip(starts, ends)
        if end - start >= 2
    ]


def _build_development_features(
    candles: DataFrame, interval: pd.Timedelta
) -> DataFrame:
    segments = _contiguous_segments(candles, interval)
    if not segments:
        raise ValueError("rl_development_data_has_no_contiguous_segment")
    feature_frames: list[DataFrame] = []
    for segment in segments:
        atr = calculate_atr(segment, expected_interval=interval)
        signal, _ = generate_donchian_breakout_signal(segment)
        features = segment.copy()
        features["atr_fraction"] = atr.to_numpy(dtype=np.float64) / features[
            "close"
        ].to_numpy(dtype=np.float64)
        features["primary_buy_signal"] = signal.to_numpy(dtype=np.bool_)
        feature_frames.append(features.loc[features["atr_fraction"].notna()].copy())
    combined = pd.concat(feature_frames, ignore_index=True)
    if (
        combined.empty
        or not np.isfinite(combined["atr_fraction"].to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError("rl_causal_feature_matrix_invalid")
    return combined


def _split_asset_features(dataset: AssetDataset, timeframe: str) -> AssetFeatures:
    minutes = TIMEFRAME_MINUTES[timeframe]
    interval = pd.Timedelta(minutes=minutes)
    bars = resample_complete_bars(dataset.candles, timeframe)
    train_boundary = int(len(bars) * 0.70)
    holdout_boundary = int(len(bars) * 0.85)
    train_cutoff = pd.Timestamp(bars["date"].iloc[train_boundary])
    validation_cutoff = pd.Timestamp(bars["date"].iloc[holdout_boundary])

    # The final 15% is not sent to ATR, Donchian, training, or validation logic.
    development_bars = bars.iloc[:holdout_boundary].reset_index(drop=True)
    features = _build_development_features(development_bars, interval)
    train_rows = features.loc[features["date"] < train_cutoff].copy()
    validation_rows = features.loc[
        (features["date"] >= train_cutoff) & (features["date"] < validation_cutoff)
    ].copy()
    train_segments = tuple(
        segment
        for segment in _contiguous_segments(train_rows, interval)
        if len(segment) >= 128
    )
    validation_segments = tuple(
        segment
        for segment in _contiguous_segments(validation_rows, interval)
        if len(segment) >= 2
    )
    if not train_segments or not validation_segments:
        raise ValueError(f"rl_temporal_split_insufficient:{dataset.symbol}")
    return AssetFeatures(
        symbol=dataset.symbol,
        train_segments=train_segments,
        validation_segments=validation_segments,
        manifest=dataset.manifest,
    )


def _environment_arrays(
    segment: DataFrame,
) -> tuple[Float64Array, Float64Array, NDArray[np.bool_]]:
    epoch_ms = (
        pd.to_datetime(segment["date"], utc=True).astype("int64").to_numpy()
        // 1_000_000
    )
    market = np.column_stack(
        (
            epoch_ms,
            segment["open"].to_numpy(dtype=np.float64),
            segment["high"].to_numpy(dtype=np.float64),
            segment["low"].to_numpy(dtype=np.float64),
            segment["close"].to_numpy(dtype=np.float64),
            segment["volume"].to_numpy(dtype=np.float64),
        )
    ).astype(np.float64)
    atr = segment["atr_fraction"].to_numpy(dtype=np.float64)
    signal = segment["primary_buy_signal"].to_numpy(dtype=np.bool_)
    return market, atr, signal


def _create_environment(segment: DataFrame, timeframe: str) -> RiskAllocationEnv:
    market, atr, signal = _environment_arrays(segment)
    return RiskAllocationEnv(
        market,
        atr,
        signal,
        bar_interval_ms=TIMEFRAME_MINUTES[timeframe] * 60_000,
    )


def _annualized_risk_ratios(
    equity_curve: list[float], timeframe: str
) -> tuple[float | None, float | None]:
    values = np.asarray(equity_curve, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 3
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
    ):
        return None, None
    period_returns = np.diff(values) / values[:-1]
    periods_per_year = 365.0 * 24.0 * 60.0 / TIMEFRAME_MINUTES[timeframe]
    annualization = math.sqrt(periods_per_year)
    mean_return = float(np.mean(period_returns))
    volatility = float(np.std(period_returns, ddof=1))
    sharpe = annualization * mean_return / volatility if volatility > 0.0 else None
    downside = np.minimum(period_returns, 0.0)
    downside_deviation = math.sqrt(float(np.mean(np.square(downside))))
    sortino = (
        annualization * mean_return / downside_deviation
        if downside_deviation > 0.0
        else None
    )
    return sharpe, sortino


def _evaluate_validation_segment(
    model: Any | None, segment: DataFrame, timeframe: str, seed: int
) -> dict[str, Any]:
    env = _create_environment(segment, timeframe)
    observation, _ = env.reset(seed=seed)
    rewards = 0.0
    info: dict[str, Any] = {}
    equity_curve = [env.initial_equity]
    target_exposures: list[float] = []
    terminated = False
    bar_index = 0
    fixed_action = ACTION_FRACTIONS.index(FIXED_BASELINE_EXPOSURE_FRACTION)
    while not terminated:
        if model is None:
            action_index = (
                fixed_action
                if bool(segment["primary_buy_signal"].iloc[bar_index])
                else 0
            )
        else:
            action, _ = model.predict(observation, deterministic=True)
            action_index = int(np.asarray(action).item())
        observation, reward, terminated, truncated, info = env.step(action_index)
        if truncated:
            raise RuntimeError("rl_validation_unexpected_truncation")
        rewards += reward
        equity_curve.append(float(info["equity"]))
        target_exposures.append(float(info["applied_target_fraction"]))
        bar_index += 1

    sharpe, sortino = _annualized_risk_ratios(equity_curve, timeframe)
    exposure_values = np.asarray(target_exposures, dtype=np.float64)
    exposure_distribution = {
        f"{fraction:.2f}": float(np.mean(np.isclose(exposure_values, fraction)))
        for fraction in ACTION_FRACTIONS
    }
    total_traded_quote = float(info["total_traded_quote"])
    return {
        "bars": len(segment),
        "equity_observations": len(equity_curve) - 1,
        "final_equity": float(info["equity"]),
        "net_return_fraction": float(info["equity"]) / env.initial_equity - 1.0,
        "max_drawdown_fraction": float(info["max_drawdown"]),
        "annualized_sharpe_ratio": sharpe,
        "annualized_sortino_ratio_mar_zero": sortino,
        "mean_target_exposure_fraction": float(np.mean(exposure_values)),
        "max_target_exposure_fraction": float(np.max(exposure_values)),
        "time_in_market_fraction": float(np.mean(exposure_values > 0.0)),
        "target_exposure_distribution": exposure_distribution,
        "position_adjustments": int(info["position_adjustments"]),
        "total_traded_quote": total_traded_quote,
        "turnover_fraction_of_initial_equity": total_traded_quote / env.initial_equity,
        "total_trading_cost_quote": float(info["total_cost_quote"]),
        "cumulative_reward": float(rewards),
    }


def _train_ppo(
    training_segments: list[DataFrame],
    validation_segments: dict[str, list[DataFrame]],
    *,
    timeframe: str,
    timesteps: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    sb3 = importlib.import_module("stable_baselines3")
    vec_module = importlib.import_module("stable_baselines3.common.vec_env")
    dummy_vec_env = getattr(vec_module, "DummyVecEnv")
    environment_factories = [
        (lambda selected=segment: _create_environment(selected, timeframe))
        for segment in training_segments
    ]
    vector_environment = dummy_vec_env(environment_factories)
    model_class = getattr(sb3, "PPO")
    model = model_class(
        "MlpPolicy",
        vector_environment,
        n_steps=128,
        batch_size=128,
        learning_rate=3e-4,
        gamma=0.99,
        seed=seed,
        verbose=0,
        device="cpu",
    )
    try:
        model.learn(total_timesteps=timesteps, progress_bar=False)
    finally:
        vector_environment.close()
    model.save(str(output_dir / "risk_allocation_ppo"))

    metrics: dict[str, list[dict[str, Any]]] = {}
    fixed_baseline_metrics: dict[str, list[dict[str, Any]]] = {}
    for asset_index, (symbol, segments) in enumerate(validation_segments.items()):
        metrics[symbol] = [
            _evaluate_validation_segment(
                model,
                segment,
                timeframe,
                seed + asset_index + segment_index,
            )
            for segment_index, segment in enumerate(segments)
        ]
        fixed_baseline_metrics[symbol] = [
            _evaluate_validation_segment(
                None,
                segment,
                timeframe,
                seed + asset_index + segment_index,
            )
            for segment_index, segment in enumerate(segments)
        ]
    return {
        "algorithm": "PPO",
        "stable_baselines3_version": str(getattr(sb3, "__version__", "unknown")),
        "training_timesteps": timesteps,
        "seed": seed,
        "model_file": "risk_allocation_ppo.zip",
        "validation_metrics": metrics,
        "fixed_position_baseline_policy": {
            "name": "PRIMARY_FULL_ALLOCATION",
            "target_exposure_fraction": FIXED_BASELINE_EXPOSURE_FRACTION,
            "primary_buy_behavior": "SET_TARGET_TO_100_PERCENT",
            "no_primary_buy_behavior": "SET_TARGET_TO_0_PERCENT",
            "purpose": "COMPARISON_ONLY_NOT_TRADE_AUTHORIZATION",
        },
        "fixed_position_baseline_metrics": fixed_baseline_metrics,
        "performance_metric_conventions": {
            "return_basis": "NET_EQUITY_PERIOD_RETURNS_AFTER_FEES_AND_SLIPPAGE",
            "annualization_basis": "365_DAY_CRYPTO_YEAR",
            "risk_free_rate_per_period": 0.0,
            "sortino_minimum_acceptable_return_per_period": 0.0,
            "sortino_downside_deviation": "RMS_OF_NEGATIVE_PERIOD_RETURNS_WITH_ZERO_FOR_NONNEGATIVE",
            "final_holdout_used": False,
        },
    }


def _validate_training_timesteps(timesteps: int) -> None:
    if isinstance(timesteps, bool) or not isinstance(timesteps, int):
        raise ValueError("rl_training_timesteps_must_be_at_least_10000")
    if timesteps < 10_000:
        raise ValueError("rl_training_timesteps_must_be_at_least_10000")


def _validate_training_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("rl_seed_must_be_non_negative_integer")
    if seed < 0:
        raise ValueError("rl_seed_must_be_non_negative_integer")


def _validate_training_request(
    timeframe: str, timesteps: int, seed: int, commit_sha: str
) -> None:
    _validate_commit_sha(commit_sha)
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError("rl_entry_timeframe_not_allowed")
    _validate_training_timesteps(timesteps)
    _validate_training_seed(seed)


def _load_training_datasets(input_dir: Path) -> dict[str, AssetDataset]:
    return {symbol: load_asset_dataset(input_dir, symbol) for symbol in ASSETS}


def _validate_common_dataset_window(datasets: dict[str, AssetDataset]) -> None:
    windows = {
        (dataset.manifest["start_utc"], dataset.manifest["end_utc"])
        for dataset in datasets.values()
    }
    if len(windows) != 1:
        raise ValueError("rl_assets_must_share_identical_dataset_window")


def _prepare_training_segments(
    datasets: dict[str, AssetDataset], timeframe: str
) -> tuple[list[DataFrame], dict[str, list[DataFrame]], dict[str, AssetFeatures]]:
    features = {
        symbol: _split_asset_features(dataset, timeframe)
        for symbol, dataset in datasets.items()
    }
    training_segments = [
        segment for feature in features.values() for segment in feature.train_segments
    ]
    validation_segments = {
        symbol: list(feature.validation_segments)
        for symbol, feature in features.items()
    }
    if len(training_segments) < 2:
        raise ValueError("rl_training_requires_multiple_contiguous_segments")
    return training_segments, validation_segments, features


def _build_training_report(
    *,
    commit_sha: str,
    timeframe: str,
    features: dict[str, AssetFeatures],
    model_results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "alix.rl-risk-allocation-ppo.v1",
        "commit_sha": commit_sha,
        "research_scope": "RISK_ALLOCATION_ONLY_NOT_PRIMARY_SIGNAL_GENERATION",
        "dataset": {
            symbol: {
                "provider": feature.manifest["provider"],
                "dataset_commit_sha": feature.manifest["commit_sha"],
                "start_utc": feature.manifest["start_utc"],
                "end_utc": feature.manifest["end_utc"],
                "requested_days": REQUESTED_DAYS,
                "row_count": feature.manifest["row_count"],
                "csv_sha256": feature.manifest["csv_sha256"],
            }
            for symbol, feature in features.items()
        },
        "entry_timeframe": timeframe,
        "allowed_entry_timeframes": list(TIMEFRAME_MINUTES),
        "primary_signal": "EXTERNAL_DONCHIAN_20_BUY_ONLY",
        "observation_fields": [
            "atr_fraction",
            "drawdown_fraction",
            "current_position_fraction",
            "primary_buy_signal",
        ],
        "actions_target_exposure_fraction": [0.0, 0.25, 0.5, 0.75, 1.0],
        "no_primary_signal_policy": "MAY_MAINTAIN_OR_REDUCE_EXPOSURE; MAY_NOT_INCREASE",
        "reward": {
            "type": "RUNNING_SORTINO_STYLE_RETURN_OVER_DOWNSIDE_RMS",
            "downside_deviation": "RUNNING_RMS_OF_NEGATIVE_LOG_RETURNS",
            "minimum_downside_deviation_fraction": 0.001,
            "reward_component_clip": 5.0,
            "drawdown_penalties_applied": True,
            "is_full_period_sortino_metric": False,
        },
        "execution_assumptions": {
            "fee_bps_per_side": 10.0,
            "slippage_bps_per_side": 5.0,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "fill_timing": "DECISION_AT_CLOSE_T_FILL_AT_NEXT_BAR_OPEN",
            "long_only": True,
            "leverage": False,
        },
        "temporal_split": {
            "training_fraction": 0.70,
            "validation_fraction": 0.15,
            "final_holdout_fraction": 0.15,
            "final_holdout_status": "RESERVED_NOT_USED_FOR_FEATURES_TRAINING_OR_TUNING",
        },
        "model": model_results,
        "economic_acceptance": "NOT_EVALUATED",
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "nobitex_api_calls": 0,
        "orders_or_execution_api_used": False,
    }


def _write_training_report(report: dict[str, Any], output_dir: Path) -> None:
    report_path = output_dir / "risk-allocation-ppo-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_training(
    *,
    input_dir: Path,
    output_dir: Path,
    timeframe: str,
    timesteps: int,
    seed: int,
    commit_sha: str,
) -> dict[str, Any]:
    _validate_training_request(timeframe, timesteps, seed, commit_sha)
    datasets = _load_training_datasets(input_dir)
    _validate_common_dataset_window(datasets)
    training_segments, validation_segments, features = _prepare_training_segments(
        datasets, timeframe
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_results = _train_ppo(
        training_segments,
        validation_segments,
        timeframe=timeframe,
        timesteps=timesteps,
        seed=seed,
        output_dir=output_dir,
    )
    report = _build_training_report(
        commit_sha=commit_sha,
        timeframe=timeframe,
        features=features,
        model_results=model_results,
    )
    _write_training_report(report, output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeframe", choices=tuple(TIMEFRAME_MINUTES), default="1h")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    report = run_training(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe,
        timesteps=args.timesteps,
        seed=args.seed,
        commit_sha=args.commit_sha,
    )
    print(
        json.dumps(
            {
                "status": "RISK_ALLOCATION_RESEARCH_TRAINED",
                "commit_sha": report["commit_sha"],
                "timeframe": report["entry_timeframe"],
                "round_trip_cost_bps": report["execution_assumptions"][
                    "round_trip_cost_bps"
                ],
                "economic_acceptance": report["economic_acceptance"],
                "Champion": report["Champion"],
                "order_permission": report["order_permission"],
                "trading_authorization": report["trading_authorization"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
