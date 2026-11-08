"""V4-AP: preregistered 36-cell Primary-only economic diagnostic sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from scripts import fetch_binance_public_ohlcv as market_fetcher
from services.ml_core.economic_backtest import (
    ExecutionConfig,
    TradeManagementConfig,
    resample_ohlcv,
    simulate_signals,
)

SCHEMA = "alix.ml-core.v4ap-diagnostic-sweep.v1"
DAYS = 1825
ASSETS = ("ETHUSDT", "BTCUSDT")
TIMEFRAMES = ("15m", "1h", "4h")
TIMEFRAME_MINUTES = {"15m": 15, "1h": 60, "4h": 240}
STRATEGIES = (
    "MEAN_REVERSION_RSI_LT_30_CLOSE_GT_EMA200",
    "DONCHIAN20_CLOSE_BREAKOUT",
)
EXIT_GEOMETRIES: tuple[dict[str, Any], ...] = (
    {
        "name": "TIGHT",
        "stop_loss_atr": 1.0,
        "take_profit_atr": 1.5,
        "trailing_enabled": False,
        "trailing_activation_atr": None,
        "trailing_step_atr": None,
    },
    {
        "name": "BALANCED",
        "stop_loss_atr": 1.5,
        "take_profit_atr": 3.0,
        "trailing_enabled": False,
        "trailing_activation_atr": None,
        "trailing_step_atr": None,
    },
    {
        "name": "TRAILING",
        "stop_loss_atr": 2.0,
        "take_profit_atr": 10.0,
        "trailing_enabled": True,
        "trailing_activation_atr": 1.0,
        "trailing_step_atr": 1.5,
    },
)
INITIAL_EQUITY_USD = 10_000.0
FEE_BPS_PER_SIDE = 10.0
SLIPPAGE_BPS_PER_SIDE = 5.0
ROUND_TRIP_COST_BPS = 30.0
ATR_PERIOD = 14
RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 30.0
EMA_TREND_PERIOD = 200
DONCHIAN_LOOKBACK = 20
HORIZON_HOURS = 24
MINIMUM_TRADE_COUNT = 50
MINIMUM_WIN_RATE = 0.40
MINIMUM_NET_PROFIT_FACTOR = 1.10
MINIMUM_DSR_Z = 0.0
DSR_TRIAL_COUNT = len(ASSETS) * len(TIMEFRAMES) * len(STRATEGIES) * len(EXIT_GEOMETRIES)
DSR_MINIMUM_RETURN_OBSERVATIONS = 4
KNOWN_SPOT_GAP_START_UTC = "2023-03-24T12:40:00Z"
KNOWN_SPOT_GAP_END_UTC = "2023-03-24T13:59:00Z"
KNOWN_SPOT_GAP_PREVIOUS_CANDLE_UTC = "2023-03-24T12:39:00Z"
KNOWN_SPOT_GAP_NEXT_CANDLE_UTC = "2023-03-24T14:00:00Z"
CSV_COLUMNS = ("date", "open", "high", "low", "close", "volume")
CONCLUSION_NO_EDGE = "CONCLUSION: NO STATISTICAL EDGE FOUND IN CURRENT FEATURE SET."
CONCLUSION_EDGE_FOUND = "CONCLUSION: EDGE CANDIDATE FOUND IN CURRENT FEATURE SET."


def _utc_timestamp(value: str, field_name: str) -> pd.Timestamp:
    """Parse an explicit timezone-aware timestamp and normalize it to UTC."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"v4ap_{field_name}_invalid") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"v4ap_{field_name}_timezone_required")
    return timestamp.tz_convert("UTC")


def _iso_utc(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _validate_request(
    commit_sha: str, start_utc: str, end_utc: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    valid_sha = len(commit_sha) == 40 and all(
        character in "0123456789abcdef" for character in commit_sha
    )
    if not valid_sha:
        raise ValueError("v4ap_commit_sha_invalid")
    start = _utc_timestamp(start_utc, "start_utc")
    end = _utc_timestamp(end_utc, "end_utc")
    if start.second or start.microsecond or end.second or end.microsecond:
        raise ValueError("v4ap_source_window_must_be_minute_aligned")
    expected_duration = pd.Timedelta(days=DAYS) - pd.Timedelta(minutes=1)
    if end - start != expected_duration:
        raise ValueError("v4ap_source_window_must_cover_exact_1825_days")
    return start, end


def _file_sha256(path: Path) -> str:
    """Hash a large market-data file without loading the entire file into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    symbol: str,
    commit_sha: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    csv_digest: str,
) -> int:
    """Fail closed on provenance, window, cost-independent safety, and gap policy."""
    expected_rows = DAYS * 24 * 60
    missing = manifest.get("missing_minute_intervals")
    if isinstance(missing, bool) or not isinstance(missing, int):
        raise ValueError(f"v4ap_market_manifest_invalid_missing_count:{symbol}")
    expected_missing = (0, 80) if symbol in ASSETS else (0,)
    expected_fields = (
        manifest.get("schema_version") == market_fetcher.SCHEMA_VERSION,
        manifest.get("provider") == "Binance Spot public klines API",
        manifest.get("symbol") == symbol,
        manifest.get("pair") == f"{symbol[:-4]}/USDT",
        manifest.get("interval") == "1m",
        manifest.get("requested_days") == DAYS,
        manifest.get("start_utc") == _iso_utc(start_utc),
        manifest.get("end_utc") == _iso_utc(end_utc),
        manifest.get("row_count") == expected_rows - missing,
        missing in expected_missing,
        manifest.get("csv_sha256") == csv_digest,
        manifest.get("commit_sha") == commit_sha,
        manifest.get("public_api_only") is True,
        manifest.get("credentials_used") is False,
        manifest.get("private_api_calls") == 0,
        manifest.get("real_order_attempts") == 0,
        manifest.get("exchange_calls") == 0,
        manifest.get("Champion") is None,
        manifest.get("order_permission") is False,
        manifest.get("trading_authorization") == "NO-GO",
    )
    if not all(expected_fields):
        raise ValueError(f"v4ap_market_manifest_contract_invalid:{symbol}")
    gap_ranges = manifest.get("missing_gap_ranges")
    if missing == 0:
        if gap_ranges != []:
            raise ValueError(f"v4ap_unexpected_gap_metadata:{symbol}")
    else:
        expected_gap = {
            "start_utc": KNOWN_SPOT_GAP_START_UTC,
            "end_utc": KNOWN_SPOT_GAP_END_UTC,
            "missing_minute_intervals": 80,
        }
        if symbol not in ASSETS or gap_ranges != [expected_gap]:
            raise ValueError("v4ap_market_gap_manifest_out_of_policy")
    return missing


def _validate_minute_timestamp_bounds(
    dates: Series,
    *,
    symbol: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> None:
    """Validate the ordered minute series and its exact inclusive UTC bounds."""
    if dates.empty or dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"v4ap_market_timestamps_invalid:{symbol}")
    if not dates.is_monotonic_increasing:
        raise ValueError(f"v4ap_market_timestamps_not_monotonic:{symbol}")
    if pd.Timestamp(dates.iloc[0]) != start_utc:
        raise ValueError(f"v4ap_market_start_mismatch:{symbol}")
    if pd.Timestamp(dates.iloc[-1]) != end_utc:
        raise ValueError(f"v4ap_market_end_mismatch:{symbol}")


def _validate_documented_spot_gap(
    dates: Series, invalid_intervals: Series, symbol: str
) -> None:
    """Accept only the one observed 80-minute Binance Spot gap for sweep assets."""
    if symbol not in ASSETS or len(invalid_intervals) != 1:
        raise ValueError("v4ap_market_gap_out_of_policy")
    gap_index = int(invalid_intervals.index[0])
    expected_previous = pd.Timestamp(KNOWN_SPOT_GAP_PREVIOUS_CANDLE_UTC)
    expected_next = pd.Timestamp(KNOWN_SPOT_GAP_NEXT_CANDLE_UTC)
    if (
        invalid_intervals.iloc[0] != pd.Timedelta(minutes=81)
        or pd.Timestamp(dates.iloc[gap_index - 1]) != expected_previous
        or pd.Timestamp(dates.iloc[gap_index]) != expected_next
    ):
        raise ValueError("v4ap_market_gap_timestamp_mismatch")


def _validate_minute_cadence(
    dates: Series,
    *,
    symbol: str,
    missing_minutes: int,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> None:
    """Accept continuous data or the exact documented 80-minute two-asset gap."""
    _validate_minute_timestamp_bounds(
        dates, symbol=symbol, start_utc=start_utc, end_utc=end_utc
    )
    differences = dates.diff().dropna()
    invalid_intervals = differences.loc[differences != pd.Timedelta(minutes=1)]
    if missing_minutes == 0:
        if not invalid_intervals.empty:
            raise ValueError(f"v4ap_undocumented_market_gap:{symbol}")
        return
    _validate_documented_spot_gap(dates, invalid_intervals, symbol)


def _load_market_data(
    input_dir: Path,
    *,
    symbol: str,
    commit_sha: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[DataFrame, dict[str, Any]]:
    """Load byte-verified public 1m data and preserve both sides of the source gap."""
    stem = f"binance-{symbol.lower()}-1m"
    csv_path = input_dir / f"{stem}.csv"
    manifest_path = input_dir / f"{stem}.manifest.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"v4ap_market_artifact_missing:{symbol}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    csv_digest = _file_sha256(csv_path)
    missing_minutes = _validate_manifest(
        manifest,
        symbol=symbol,
        commit_sha=commit_sha,
        start_utc=start_utc,
        end_utc=end_utc,
        csv_digest=csv_digest,
    )
    candles = pd.read_csv(csv_path)
    if tuple(candles.columns) != CSV_COLUMNS:
        raise ValueError(f"v4ap_market_csv_schema_invalid:{symbol}")
    candles["date"] = pd.to_datetime(candles["date"], utc=True, errors="coerce")
    _validate_minute_cadence(
        candles["date"],
        symbol=symbol,
        missing_minutes=missing_minutes,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    if len(candles) != int(manifest["row_count"]):
        raise ValueError(f"v4ap_market_csv_row_count_mismatch:{symbol}")
    for column in ("open", "high", "low", "close", "volume"):
        candles[column] = pd.to_numeric(candles[column], errors="coerce")
    numeric = candles.loc[:, ["open", "high", "low", "close", "volume"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"v4ap_market_values_invalid:{symbol}")
    provenance = {
        "symbol": symbol,
        "provider": manifest["provider"],
        "interval": manifest["interval"],
        "requested_days": manifest["requested_days"],
        "start_utc": manifest["start_utc"],
        "end_utc": manifest["end_utc"],
        "row_count": manifest["row_count"],
        "missing_minute_intervals": missing_minutes,
        "missing_gap_ranges": manifest["missing_gap_ranges"],
        "csv_sha256": csv_digest,
        "commit_sha": manifest["commit_sha"],
        "public_api_only": True,
        "credentials_used": False,
    }
    return candles.loc[:, list(CSV_COLUMNS)], provenance


def _causal_wilder_rsi(close: Series, period: int = RSI_PERIOD) -> Series:
    """Compute Wilder RSI with only current and previously completed closes."""
    changes = close.diff()
    gains = changes.clip(lower=0.0)
    losses = -changes.clip(upper=0.0)
    average_gain = gains.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    average_loss = losses.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    movement = average_gain + average_loss
    rsi = 100.0 * average_gain / movement.where(movement > 0.0)
    return rsi.mask(movement.eq(0.0), 50.0)


def generate_mean_reversion_signal(bars: DataFrame) -> tuple[Series, dict[str, Any]]:
    """Generate every completed-bar RSI<30 and Close>EMA200 spot-long signal."""
    if not {"close"}.issubset(bars.columns) or bars.empty:
        raise ValueError("v4ap_mean_reversion_bars_invalid")
    close = pd.to_numeric(bars["close"], errors="coerce").reset_index(drop=True)
    values = close.to_numpy(dtype=float)
    if not np.isfinite(values).all() or bool((close <= 0.0).any()):
        raise ValueError("v4ap_mean_reversion_close_invalid")
    rsi = _causal_wilder_rsi(close)
    ema_200 = close.ewm(
        span=EMA_TREND_PERIOD,
        adjust=False,
        min_periods=EMA_TREND_PERIOD,
    ).mean()
    signal = (
        (
            rsi.lt(RSI_ENTRY_THRESHOLD)
            & close.gt(ema_200)
            & Series(np.isfinite(rsi.to_numpy(dtype=float)))
            & Series(np.isfinite(ema_200.to_numpy(dtype=float)))
        )
        .fillna(False)
        .astype(bool)
    )
    signal.name = "primary_buy_signal"
    diagnostics = {
        "rsi_period": RSI_PERIOD,
        "rsi_method": "WILDER_EMA_ALPHA_1_OVER_PERIOD_ADJUST_FALSE",
        "rsi_threshold": RSI_ENTRY_THRESHOLD,
        "rsi_comparison": "STRICT_LT",
        "ema_period": EMA_TREND_PERIOD,
        "ema_method": "EMA_SPAN_ADJUST_FALSE",
        "trend_comparison": "CLOSE_STRICTLY_GREATER_THAN_EMA200",
        "primary_signal_count": int(signal.sum()),
        "lookahead_used": False,
        "hmm_used": False,
        "cusum_used": False,
    }
    return signal, diagnostics


def generate_donchian_breakout_signal(
    bars: DataFrame,
) -> tuple[Series, dict[str, Any]]:
    """Buy when close exceeds the highest high of the previous 20 closed bars."""
    if not {"high", "close"}.issubset(bars.columns) or bars.empty:
        raise ValueError("v4ap_donchian_bars_invalid")
    high = pd.to_numeric(bars["high"], errors="coerce").reset_index(drop=True)
    close = pd.to_numeric(bars["close"], errors="coerce").reset_index(drop=True)
    if not np.isfinite(high.to_numpy(dtype=float)).all() or bool((high <= 0.0).any()):
        raise ValueError("v4ap_donchian_high_invalid")
    if not np.isfinite(close.to_numpy(dtype=float)).all() or bool((close <= 0.0).any()):
        raise ValueError("v4ap_donchian_close_invalid")
    prior_20_high = (
        high.shift(1)
        .rolling(window=DONCHIAN_LOOKBACK, min_periods=DONCHIAN_LOOKBACK)
        .max()
    )
    signal = (
        (close.gt(prior_20_high) & prior_20_high.notna()).fillna(False).astype(bool)
    )
    signal.name = "primary_buy_signal"
    diagnostics = {
        "lookback_bars": DONCHIAN_LOOKBACK,
        "reference": "HIGHEST_HIGH_OF_PREVIOUS_20_COMPLETED_BARS",
        "comparison": "CLOSE_STRICTLY_GREATER_THAN_PRIOR_20_BAR_HIGH",
        "primary_signal_count": int(signal.sum()),
        "lookahead_used": False,
        "hmm_used": False,
        "cusum_used": False,
    }
    return signal, diagnostics


def _validate_signal_execution_inputs(
    bars: DataFrame,
    primary_signal: Series,
    timeframe: str,
    horizon_bars: int,
) -> None:
    """Fail closed when signal rows or execution horizon are not aligned."""
    if len(bars) != len(primary_signal) or len(bars) < 2:
        raise ValueError("v4ap_signal_bar_alignment_invalid")
    if timeframe not in TIMEFRAME_MINUTES or horizon_bars < 1:
        raise ValueError("v4ap_signal_execution_policy_invalid")


def _execution_action(
    signal_timestamp: pd.Timestamp,
    execution_timestamp: pd.Timestamp,
    action: str,
) -> dict[str, Any]:
    """Create a deterministic next-bar execution instruction."""
    return {
        "signal_timestamp": signal_timestamp.isoformat(),
        "execution_timestamp": execution_timestamp.isoformat(),
        "signal": action,
        "buy_probability": 1.0,
        "position_fraction": 1.0,
    }


def _entry_signal_is_eligible(
    signal_index: int, last_index: int, horizon_bars: int
) -> bool:
    """Require a full vertical-barrier horizon before the dataset boundary."""
    last_entry_signal_index = last_index - horizon_bars
    return signal_index <= last_entry_signal_index


def _build_execution_signals(
    bars: DataFrame,
    primary_signal: Series,
    *,
    timeframe: str,
    horizon_bars: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Map closed-bar signals to next opens and flatten across data discontinuities."""
    _validate_signal_execution_inputs(bars, primary_signal, timeframe, horizon_bars)
    dates = pd.to_datetime(bars["date"], utc=True).reset_index(drop=True)
    expected_delta = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    signal_values = primary_signal.to_numpy(dtype=bool)
    eligible_entry_count = 0
    gap_flatten_count = 0
    execution_signals: list[dict[str, Any]] = []
    last_index = len(bars) - 1
    for index in range(last_index):
        next_date = pd.Timestamp(dates.iloc[index + 1])
        current_date = pd.Timestamp(dates.iloc[index])
        is_gap_boundary = next_date - current_date != expected_delta
        if is_gap_boundary:
            execution_signals.append(_execution_action(current_date, next_date, "SELL"))
            gap_flatten_count += 1
            continue
        if not _entry_signal_is_eligible(index, last_index, horizon_bars):
            continue
        if not signal_values[index]:
            continue
        execution_signals.append(_execution_action(current_date, next_date, "BUY"))
        eligible_entry_count += 1
    if last_index > 0:
        execution_signals.append(
            _execution_action(
                pd.Timestamp(dates.iloc[-2]),
                pd.Timestamp(dates.iloc[-1]),
                "NO_ENTRY",
            )
        )
    execution_signals.sort(key=lambda item: item["execution_timestamp"])
    if not execution_signals:
        raise ValueError("v4ap_execution_signal_set_empty")
    return execution_signals, eligible_entry_count, gap_flatten_count


def _trade_return_diagnostics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate per-trade Sharpe and higher moments for DSR, without annualizing."""
    returns = np.asarray(
        [float(item["net_return_fraction"]) for item in trades], dtype=float
    )
    if returns.ndim != 1 or not np.isfinite(returns).all():
        raise ValueError("v4ap_trade_returns_invalid")
    count = int(returns.size)
    result: dict[str, Any] = {
        "return_observation_count": count,
        "sharpe_ratio_per_trade": None,
        "return_skewness": None,
        "return_kurtosis_non_excess": None,
        "sharpe_status": ("NO_TRADES_DSR_UNDEFINED" if count == 0 else "INSUFFICIENT_TRADE_RETURNS"),
    }
    if count < 2:
        return result
    standard_deviation = float(np.std(returns, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        result["sharpe_status"] = "ZERO_RETURN_VOLATILITY"
        return result
    observed_sharpe = float(np.mean(returns) / standard_deviation)
    result["sharpe_ratio_per_trade"] = observed_sharpe
    result["sharpe_status"] = "COMPUTED"
    if count < DSR_MINIMUM_RETURN_OBSERVATIONS:
        return result
    centered = returns - float(np.mean(returns))
    second_moment = float(np.mean(np.square(centered)))
    if second_moment <= 0.0 or not math.isfinite(second_moment):
        result["sharpe_status"] = "RETURN_DISTRIBUTION_UNAVAILABLE"
        return result
    third_moment = float(np.mean(np.power(centered, 3)))
    fourth_moment = float(np.mean(np.power(centered, 4)))
    biased_skew = third_moment / second_moment**1.5
    adjusted_skew = math.sqrt(count * (count - 1)) / (count - 2) * biased_skew
    biased_excess_kurtosis = fourth_moment / second_moment**2 - 3.0
    adjusted_excess_kurtosis = (
        (count - 1)
        / ((count - 2) * (count - 3))
        * ((count + 1) * biased_excess_kurtosis + 6.0)
    )
    result["return_skewness"] = float(adjusted_skew)
    result["return_kurtosis_non_excess"] = float(adjusted_excess_kurtosis + 3.0)
    return result


def _expected_maximum_sharpe(
    valid_trial_sharpes: list[float], total_trial_count: int
) -> float:
    """Use valid Sharpe estimates but deflate against the full preregistered N=36."""
    trial_values = np.asarray(valid_trial_sharpes, dtype=float)
    if total_trial_count < 2 or trial_values.size < 2:
        raise ValueError("v4ap_dsr_trial_sharpe_support_insufficient")
    if not np.isfinite(trial_values).all():
        raise ValueError("v4ap_dsr_trial_sharpe_distribution_invalid")
    euler_mascheroni = 0.5772156649015329
    normal = NormalDist()
    expected_max_multiplier = (1.0 - euler_mascheroni) * normal.inv_cdf(
        1.0 - 1.0 / total_trial_count
    ) + euler_mascheroni * normal.inv_cdf(1.0 - 1.0 / (total_trial_count * math.e))
    return (
        float(np.mean(trial_values))
        + float(np.std(trial_values, ddof=1)) * expected_max_multiplier
    )


def _unavailable_candidate_dsr(
    valid_sharpe_count: int, trade_count: int
) -> dict[str, Any]:
    """Return undefined DSR for no-trade cells or insufficient cohort support."""
    return {
        "dsr_z_score": None,
        "dsr_probability": None,
        "dsr_trial_count": DSR_TRIAL_COUNT,
        "dsr_valid_sharpe_estimate_count": valid_sharpe_count,
        "dsr_trial_scope": "ALL_36_PREREGISTERED_COMBINATIONS",
        "dsr_status": (
            "NO_TRADES_DSR_UNDEFINED"
            if trade_count == 0
            else "INSUFFICIENT_TRIAL_SHARPE_ESTIMATES"
        ),
    }


def _candidate_dsr_diagnostics(
    item: dict[str, Any],
    expected_maximum: float,
    valid_sharpe_count: int,
) -> dict[str, Any]:
    """Calculate one candidate's deflated Sharpe result or an explicit failure state."""
    skew = item.get("return_skewness")
    kurtosis = item.get("return_kurtosis_non_excess")
    observed = item.get("sharpe_ratio_per_trade")
    count = int(item.get("return_observation_count", 0))
    if count == 0:
        dsr_status = "NO_TRADES_DSR_UNDEFINED"
        z_score = None
        probability = None
    elif (
        observed is None
        or skew is None
        or kurtosis is None
        or count < DSR_MINIMUM_RETURN_OBSERVATIONS
    ):
        dsr_status = "RETURN_DISTRIBUTION_UNAVAILABLE"
        z_score = None
        probability = None
    else:
        variance_term = (
            1.0
            - float(skew) * float(observed)
            + ((float(kurtosis) - 1.0) / 4.0) * float(observed) ** 2
        )
        if not math.isfinite(variance_term) or variance_term <= 0.0:
            dsr_status = "DSR_VARIANCE_INVALID"
            z_score = None
            probability = None
        else:
            z_score = (
                (float(observed) - expected_maximum)
                * math.sqrt(count - 1)
                / math.sqrt(variance_term)
            )
            probability = NormalDist().cdf(z_score)
            if not math.isfinite(z_score) or not math.isfinite(probability):
                dsr_status = "DSR_RESULT_NON_FINITE"
                z_score = None
                probability = None
            else:
                dsr_status = "COMPUTED"
    return {
        "dsr_z_score": z_score,
        "dsr_probability": probability,
        "dsr_expected_maximum_sharpe_per_trade": expected_maximum,
        "dsr_trial_count": DSR_TRIAL_COUNT,
        "dsr_valid_sharpe_estimate_count": valid_sharpe_count,
        "dsr_trial_scope": "ALL_36_PREREGISTERED_COMBINATIONS",
        "dsr_status": dsr_status,
    }


def _attach_dsr_to_cohort(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach DSR using the complete 36-test cohort, not only successful cells."""
    valid_sharpes = [
        float(item["sharpe_ratio_per_trade"])
        for item in candidates
        if item.get("sharpe_ratio_per_trade") is not None
        and math.isfinite(float(item["sharpe_ratio_per_trade"]))
    ]
    cohort: dict[str, Any] = {
        "trial_count": DSR_TRIAL_COUNT,
        "valid_sharpe_estimate_count": len(valid_sharpes),
        "trial_scope": "ALL_36_PREREGISTERED_COMBINATIONS",
        "independence_status": "NOT_VERIFIED; ALL_36_USED_AS_NOMINAL_TRIAL_COUNT",
        "expected_maximum_sharpe_per_trade": None,
    }
    if len(valid_sharpes) < 2:
        for item in candidates:
            item.update(
                _unavailable_candidate_dsr(
                    len(valid_sharpes), int(item.get("trade_count", 0))
                )
            )
        return cohort
    expected_maximum = _expected_maximum_sharpe(valid_sharpes, DSR_TRIAL_COUNT)
    cohort["expected_maximum_sharpe_per_trade"] = expected_maximum
    for item in candidates:
        item.update(
            _candidate_dsr_diagnostics(item, expected_maximum, len(valid_sharpes))
        )
    return cohort


def _candidate_qualifies(candidate: dict[str, Any]) -> bool:
    """Apply the owner's four strict, pre-registered acceptance dimensions."""
    profit_factor = candidate.get("net_profit_factor")
    infinite_pf = candidate.get("net_profit_factor_infinite_no_losses") is True
    win_rate = candidate.get("win_rate")
    dsr_z = candidate.get("dsr_z_score")
    return (
        int(candidate.get("trade_count", 0)) >= MINIMUM_TRADE_COUNT
        and win_rate is not None
        and float(win_rate) >= MINIMUM_WIN_RATE
        and (
            infinite_pf
            or (
                profit_factor is not None
                and float(profit_factor) >= MINIMUM_NET_PROFIT_FACTOR
            )
        )
        and dsr_z is not None
        and float(dsr_z) > MINIMUM_DSR_Z
    )


def _candidate_id(symbol: str, timeframe: str, strategy: str, geometry: str) -> str:
    return "|".join((symbol, timeframe, strategy, geometry))


def _grid_candidate_ids() -> set[str]:
    return {
        _candidate_id(symbol, timeframe, strategy, geometry["name"])
        for symbol in ASSETS
        for timeframe in TIMEFRAMES
        for strategy in STRATEGIES
        for geometry in EXIT_GEOMETRIES
    }


def _candidate_performance_fields(
    metrics: dict[str, Any], trades: list[dict[str, Any]]
) -> dict[str, Any]:
    """Normalize simulator metrics, retaining an explicit zero-trade result row."""
    trade_count = len(trades)
    if int(metrics["trade_count"]) != trade_count:
        raise ValueError("v4ap_simulator_trade_count_mismatch")
    if trade_count == 0:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "net_profit_factor": 0.0,
            "net_profit_factor_infinite_no_losses": False,
            "max_drawdown_percent": 0.0,
        }
    profit_factor = metrics.get("profit_factor_net_of_costs")
    return {
        "trade_count": trade_count,
        "win_rate": float(metrics["win_rate"]),
        "net_profit_factor": (
            float(profit_factor) if profit_factor is not None else None
        ),
        "net_profit_factor_infinite_no_losses": bool(
            metrics["profit_factor_infinite_no_losses"]
        ),
        "max_drawdown_percent": float(metrics["max_drawdown_percent"]),
    }


def _run_candidate(
    bars: DataFrame,
    *,
    symbol: str,
    timeframe: str,
    strategy: str,
    geometry: dict[str, Any],
    signal: Series,
    strategy_diagnostics: dict[str, Any],
    execution: ExecutionConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute one Primary-only cell with the shared existing economic simulator."""
    horizon_bars = HORIZON_HOURS * 60 // TIMEFRAME_MINUTES[timeframe]
    execution_signals, entry_signal_count, gap_flatten_count = _build_execution_signals(
        bars, signal, timeframe=timeframe, horizon_bars=horizon_bars
    )
    management = TradeManagementConfig(
        atr_period=ATR_PERIOD,
        take_profit_atr_multiplier=float(geometry["take_profit_atr"]),
        stop_loss_atr_multiplier=float(geometry["stop_loss_atr"]),
        horizon_candles=horizon_bars,
        confidence_threshold=None,
        trailing_stop_enabled=bool(geometry["trailing_enabled"]),
        trailing_activation_atr=float(geometry["trailing_activation_atr"] or 1.0),
        trailing_step_atr=float(geometry["trailing_step_atr"] or 1.5),
        time_exit_candles=None,
    )
    replay = simulate_signals(
        bars,
        execution_signals,
        execution,
        trade_management_config=management,
    )
    metrics = replay["metrics"]
    trades = replay["trades"]
    risk = _trade_return_diagnostics(trades)
    candidate: dict[str, Any] = {
        "candidate_id": _candidate_id(
            symbol, timeframe, strategy, str(geometry["name"])
        ),
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy,
        "geometry": str(geometry["name"]),
        "bars_evaluated": len(bars),
        "raw_signal_count": int(strategy_diagnostics["primary_signal_count"]),
        "eligible_entry_signal_count": entry_signal_count,
        "gap_flatten_count": gap_flatten_count,
        **_candidate_performance_fields(metrics, trades),
        "net_pnl_usd": float(metrics["final_equity_usd"])
        - float(metrics["initial_equity_usd"]),
        "net_expectancy_usd_per_trade": metrics["average_net_pnl_per_trade_usd"],
        "total_fees_usd": float(replay["total_fees_usd"]),
        "total_slippage_cost_usd": float(replay["total_slippage_cost_usd"]),
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "fee_bps_per_side": FEE_BPS_PER_SIDE,
        "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
        "stop_loss_atr": float(geometry["stop_loss_atr"]),
        "take_profit_atr": float(geometry["take_profit_atr"]),
        "trailing_enabled": bool(geometry["trailing_enabled"]),
        "trailing_activation_atr": geometry["trailing_activation_atr"],
        "trailing_step_atr": geometry["trailing_step_atr"],
        "horizon_hours_nominal": HORIZON_HOURS,
        "horizon_bars": horizon_bars,
        "same_candle_collision_policy": "STOP_LOSS_WINS",
        "execution_policy": "CLOSED_BAR_NEXT_BAR_OPEN",
        "strategy_diagnostics": strategy_diagnostics,
        **risk,
    }
    candidate["qualifies"] = _candidate_qualifies(candidate)
    trade_rows: list[dict[str, Any]] = []
    for trade in trades:
        trade_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy,
                "geometry": geometry["name"],
                **trade,
            }
        )
    return candidate, trade_rows


def _execution_config() -> ExecutionConfig:
    execution = ExecutionConfig(
        initial_equity=INITIAL_EQUITY_USD,
        fee_bps=FEE_BPS_PER_SIDE,
        slippage_bps=SLIPPAGE_BPS_PER_SIDE,
        position_fraction=1.0,
        maximum_drawdown=0.99,
    )
    if not math.isclose(
        2.0 * (execution.fee_bps + execution.slippage_bps),
        ROUND_TRIP_COST_BPS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("v4ap_round_trip_cost_contract_invalid")
    return execution


def _write_trade_csv(path: Path, trades: list[dict[str, Any]]) -> None:
    """Persist the aggregate trade ledger in a deterministic flat CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in trades for key in row})
    if not columns:
        columns = ["candidate_id", "symbol", "timeframe", "strategy", "geometry"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in trades:
            serialized = {
                key: (
                    json.dumps(value, sort_keys=True, allow_nan=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(serialized)


def _build_report(
    *,
    commit_sha: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    provenance: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
    dsr_cohort: dict[str, Any],
) -> dict[str, Any]:
    qualifying = [item["candidate_id"] for item in candidates if item["qualifies"]]
    conclusion = CONCLUSION_EDGE_FOUND if qualifying else CONCLUSION_NO_EDGE
    return {
        "schema_version": SCHEMA,
        "commit_sha": commit_sha,
        "diagnostic": "V4-AP_FINAL_COMPREHENSIVE_PRIMARY_ONLY_SWEEP",
        "dataset": {
            "requested_days": DAYS,
            "source_interval": "1m",
            "start_utc": _iso_utc(start_utc),
            "end_utc": _iso_utc(end_utc),
            "assets": provenance,
            "analysis_timeframes": list(TIMEFRAMES),
            "bar_aggregation": "UTC_EPOCH_ALIGNED_OHLCV_COMPLETE_BARS_ONLY",
            "missing_candles": "NEVER_INTERPOLATED_OR_SYNTHETICALLY_FILLED",
            "documented_spot_gap_policy": (
                "ONLY_THE_DOCUMENTED_80_MINUTE_BINANCE_SPOT_GAP_FOR_BTCUSDT_AND_ETHUSDT "
                "MAY_BE_ACCEPTED; ALL_OTHER_GAPS_FAIL_CLOSED"
            ),
        },
        "preregistered_grid": {
            "assets": list(ASSETS),
            "timeframes": list(TIMEFRAMES),
            "strategies": list(STRATEGIES),
            "exit_geometries": [dict(item) for item in EXIT_GEOMETRIES],
            "expected_combinations": DSR_TRIAL_COUNT,
        },
        "execution_assumptions": {
            "market": "BINANCE_SPOT_PUBLIC_OHLCV",
            "direction": "SPOT_LONG_ONLY",
            "initial_equity_usd": INITIAL_EQUITY_USD,
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "position_fraction": 1.0,
            "pyramiding": False,
            "signal_timing": "COMPLETED_BAR_CLOSE_NEXT_BAR_OPEN",
            "atr_period": ATR_PERIOD,
            "horizon_hours_nominal": HORIZON_HOURS,
            "horizon_policy": (
                "FIXED_24_HOUR_NOMINAL_HORIZON_CONVERTED_TO_COMPLETE_BARS; "
                "POSITIONS_FLATTENED_AT_DATA_DISCONTINUITIES"
            ),
            "same_candle_collision_policy": "STOP_LOSS_WINS",
            "passive_limit_queue_and_partial_fills_simulated": False,
        },
        "strategies": {
            STRATEGIES[0]: {
                "rsi_period": RSI_PERIOD,
                "rsi_method": "WILDER_EMA_ALPHA_1_OVER_PERIOD_ADJUST_FALSE",
                "rsi_condition": "RSI_STRICTLY_LESS_THAN_30",
                "ema_period": EMA_TREND_PERIOD,
                "trend_condition": "CLOSE_STRICTLY_GREATER_THAN_EMA200",
                "event_policy": "EVERY_COMPLETED_BAR_WHILE_CONDITION_IS_TRUE",
            },
            STRATEGIES[1]: {
                "donchian_lookback_bars": DONCHIAN_LOOKBACK,
                "reference": "HIGHEST_HIGH_OF_PREVIOUS_20_COMPLETED_BARS",
                "breakout_condition": "CLOSE_STRICTLY_GREATER_THAN_PRIOR_20_BAR_HIGH",
                "event_policy": "EVERY_COMPLETED_BAR_WHILE_CONDITION_IS_TRUE",
            },
        },
        "dsr_cohort": dsr_cohort,
        "decision_rule": {
            "minimum_trade_count": MINIMUM_TRADE_COUNT,
            "minimum_win_rate": MINIMUM_WIN_RATE,
            "minimum_net_profit_factor": MINIMUM_NET_PROFIT_FACTOR,
            "minimum_dsr_z_exclusive": MINIMUM_DSR_Z,
            "profit_factor_comparison": "GREATER_THAN_OR_EQUAL",
            "win_rate_comparison": "GREATER_THAN_OR_EQUAL",
            "dsr_z_comparison": "STRICTLY_GREATER_THAN",
            "multiple_testing": (
                "DSR_EXPECTED_MAXIMUM_USES_ALL_36_PREREGISTERED_TRIALS; "
                "VALID_SHARPE_ESTIMATES_SUPPLY_THE_SHARPE_DISTRIBUTION"
            ),
            "dsr_return_basis": "NET_PER_TRADE_RETURNS_NOT_ANNUALIZED",
        },
        "candidates": candidates,
        "qualifying_candidate_ids": qualifying,
        "conclusion": conclusion,
        "model_usage": {
            "meta_model_used": False,
            "hmm_used": False,
            "cusum_used": False,
            "lstm_used": False,
            "reinforcement_learning_used": False,
        },
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "credentials_used": False,
            "sequence_model_used": False,
        },
        "promotion_status": "BLOCKED_OWNER_REVIEW_AND_INDEPENDENT_VALIDATION",
    }


def _candidate_report_contract_is_valid(candidate: dict[str, Any]) -> bool:
    """Check one row's fixed costs, dimensions, and decision consistency."""
    expected = (
        ROUND_TRIP_COST_BPS,
        FEE_BPS_PER_SIDE,
        SLIPPAGE_BPS_PER_SIDE,
    )
    actual = (
        candidate.get("round_trip_cost_bps"),
        candidate.get("fee_bps_per_side"),
        candidate.get("slippage_bps_per_side"),
    )
    valid_geometry_names = {str(item["name"]) for item in EXIT_GEOMETRIES}
    return (
        actual == expected
        and candidate.get("timeframe") in TIMEFRAMES
        and candidate.get("symbol") in ASSETS
        and candidate.get("strategy") in STRATEGIES
        and candidate.get("geometry") in valid_geometry_names
        and candidate.get("qualifies") is _candidate_qualifies(candidate)
    )


def _validate_candidate_grid(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Require exactly the preregistered unique 36 candidate identities."""
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != DSR_TRIAL_COUNT:
        raise ValueError("v4ap_candidate_grid_cardinality_invalid")
    candidate_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("v4ap_candidate_row_invalid")
        candidate_ids.add(str(candidate.get("candidate_id")))
        if not _candidate_report_contract_is_valid(candidate):
            raise ValueError("v4ap_candidate_contract_invalid")
    if candidate_ids != _grid_candidate_ids():
        raise ValueError("v4ap_candidate_grid_identity_invalid")
    return candidates


def _validate_report_decision(
    report: dict[str, Any], candidates: list[dict[str, Any]]
) -> None:
    """Recompute the qualification list and exact terminal decision."""
    qualifying = [
        item["candidate_id"] for item in candidates if _candidate_qualifies(item)
    ]
    if report.get("qualifying_candidate_ids") != qualifying:
        raise ValueError("v4ap_qualifying_candidate_list_invalid")
    expected_conclusion = CONCLUSION_EDGE_FOUND if qualifying else CONCLUSION_NO_EDGE
    if report.get("conclusion") != expected_conclusion:
        raise ValueError("v4ap_final_conclusion_invalid")


def _validate_report_trial_scope(report: dict[str, Any]) -> None:
    """Keep the DSR and grid correction anchored to all 36 registered trials."""
    if report.get("dsr_cohort", {}).get("trial_count") != DSR_TRIAL_COUNT:
        raise ValueError("v4ap_dsr_trial_count_invalid")
    if (
        report.get("preregistered_grid", {}).get("expected_combinations")
        != DSR_TRIAL_COUNT
    ):
        raise ValueError("v4ap_preregistered_grid_invalid")


def _validate_report_safety(report: dict[str, Any]) -> None:
    """Reject AI usage or any promotion/trading permission in the report."""
    expected_model_usage = {
        "meta_model_used": False,
        "hmm_used": False,
        "cusum_used": False,
        "lstm_used": False,
        "reinforcement_learning_used": False,
    }
    if report.get("model_usage") != expected_model_usage:
        raise ValueError("v4ap_forbidden_model_usage_detected")
    expected_safety = {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "credentials_used": False,
        "sequence_model_used": False,
    }
    if report.get("safety") != expected_safety:
        raise ValueError("v4ap_safety_contract_invalid")


def _validate_report_execution(report: dict[str, Any]) -> None:
    """Require the report to disclose and preserve the 30 bps cost assumption."""
    execution = report.get("execution_assumptions")
    if not isinstance(execution, dict):
        raise ValueError("v4ap_execution_cost_contract_invalid")
    if execution.get("round_trip_cost_bps") != ROUND_TRIP_COST_BPS:
        raise ValueError("v4ap_execution_cost_contract_invalid")


def _validate_report(report: dict[str, Any], expected_commit_sha: str) -> None:
    """Validate the preregistered grid, DSR scope, conclusion, and safety state."""
    if report.get("schema_version") != SCHEMA:
        raise ValueError("v4ap_report_schema_invalid")
    if report.get("commit_sha") != expected_commit_sha:
        raise ValueError("v4ap_report_commit_mismatch")
    candidates = _validate_candidate_grid(report)
    _validate_report_decision(report, candidates)
    _validate_report_trial_scope(report)
    _validate_report_safety(report)
    _validate_report_execution(report)


def _format_number(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{decimals}f}"


def render_step_summary(report: dict[str, Any]) -> str:
    """Render all 36 preregistered results and the exact owner decision line."""
    lines = [
        "# AliX V4-AP: Final Comprehensive Diagnostic Sweep",
        "",
        f"Commit SHA: {report['commit_sha']}",
        f"Dataset: {DAYS} requested days | 1m source | "
        f"{report['dataset']['start_utc']} through {report['dataset']['end_utc']}",
        "Assets: ETHUSDT, BTCUSDT | Timeframes: 15m, 1h, 4h",
        "Scope: Primary-only; Meta/HMM/CUSUM/LSTM/RL are not used.",
        "Execution: Spot-long only; completed-bar signal, next-bar-open entry.",
        "Costs: 10 bps fee + 5 bps slippage per side = 30 bps round trip.",
        (
            "Nominal holding horizon: 24 hours, converted to 96/24/6 complete bars "
            "for 15m/1h/4h; open positions are flattened at a data discontinuity."
        ),
        "",
        "## Preregistered acceptance rule",
        (
            "Trades >= 50; Win Rate >= 40%; Net PF >= 1.10; "
            "DSR Z > 0.0. All four conditions must pass."
        ),
        (
            "DSR uses the 36 preregistered combinations as the nominal trial count; "
            "Sharpe and DSR are per-trade, not annualized. Trial independence "
            "is not verified."
        ),
        "",
        "## Results: all 36 combinations",
        "",
        "| Asset | TF | Strategy | Exit | Signals | Trades | Win % | Net PF | Max DD % | DSR Z | Net PnL USD | Expectancy USD | Decision |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["candidates"]:
        pf = (
            "∞"
            if item["net_profit_factor_infinite_no_losses"]
            else _format_number(item["net_profit_factor"])
        )
        dsr = _format_number(item["dsr_z_score"])
        decision = "PASS" if item["qualifies"] else "NO"
        lines.append(
            "| "
            + " | ".join(
                (
                    str(item["symbol"]),
                    str(item["timeframe"]),
                    str(item["strategy"]),
                    str(item["geometry"]),
                    str(item["raw_signal_count"]),
                    str(item["trade_count"]),
                    _format_number(float(item["win_rate"]) * 100.0, 2),
                    pf,
                    _format_number(item["max_drawdown_percent"], 2),
                    dsr,
                    _format_number(item["net_pnl_usd"], 2),
                    _format_number(item["net_expectancy_usd_per_trade"], 4),
                    decision,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            str(report["conclusion"]),
        ]
    )
    if report["qualifying_candidate_ids"]:
        lines.append("")
        lines.append("Qualifying candidate(s):")
        lines.extend(f"- {item}" for item in report["qualifying_candidate_ids"])
        lines.append(
            "Re-apply Meta-Labeling only to a selected candidate after owner review; "
            "this sweep does not authorize promotion or trading."
        )
    else:
        lines.extend(
            [
                "",
                (
                    "Required pivot: evaluate approved point-in-time external data "
                    "(including funding rates and order-book imbalance) or an isolated "
                    "reinforcement-learning research track. External providers must "
                    "pass the MRD provider registry, licensing, provenance, and "
                    "Data Quality Gate requirements."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Dataset integrity",
            "",
            "| Asset | 1m rows | Missing minutes | SHA-256 |",
            "|---|---:|---:|---|",
        ]
    )
    for symbol in ASSETS:
        source = report["dataset"]["assets"][symbol]
        lines.append(
            f"| {symbol} | {source['row_count']} | "
            f"{source['missing_minute_intervals']} | {source['csv_sha256']} |"
        )
    lines.extend(
        [
            "",
            "## Safety / promotion",
            "",
            "Champion=null | order_permission=false | trading_authorization=NO-GO",
            "private_api_calls=0 | real_order_attempts=0 | credentials_used=false",
            "Promotion remains blocked pending independent review and validation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_v4ap(
    input_dir: Path,
    output_report: Path,
    output_trades_csv: Path,
    output_summary: Path,
    *,
    commit_sha: str,
    start_utc: str,
    end_utc: str,
) -> dict[str, Any]:
    start, end = _validate_request(commit_sha, start_utc, end_utc)
    execution = _execution_config()
    provenance: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    for symbol in ASSETS:
        minute_data, asset_provenance = _load_market_data(
            input_dir,
            symbol=symbol,
            commit_sha=commit_sha,
            start_utc=start,
            end_utc=end,
        )
        provenance[symbol] = asset_provenance
        for timeframe in TIMEFRAMES:
            bars = resample_ohlcv(minute_data, timeframe)
            signal_generators = {
                STRATEGIES[0]: generate_mean_reversion_signal,
                STRATEGIES[1]: generate_donchian_breakout_signal,
            }
            for strategy in STRATEGIES:
                signal, diagnostics = signal_generators[strategy](bars)
                for geometry in EXIT_GEOMETRIES:
                    candidate, trade_rows = _run_candidate(
                        bars,
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy=strategy,
                        geometry=geometry,
                        signal=signal,
                        strategy_diagnostics=diagnostics,
                        execution=execution,
                    )
                    candidates.append(candidate)
                    all_trades.extend(trade_rows)
            del bars
        del minute_data
    dsr_cohort = _attach_dsr_to_cohort(candidates)
    for candidate in candidates:
        candidate["qualifies"] = _candidate_qualifies(candidate)
    report = _build_report(
        commit_sha=commit_sha,
        start_utc=start,
        end_utc=end,
        provenance=provenance,
        candidates=candidates,
        dsr_cohort=dsr_cohort,
    )
    _validate_report(report, commit_sha)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_trade_csv(output_trades_csv, all_trades)
    summary = render_step_summary(report)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(summary, encoding="utf-8")
    print(summary, end="")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-trades-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--end-utc", required=True)
    args = parser.parse_args()
    report = run_v4ap(
        args.input_dir,
        args.output_report,
        args.output_trades_csv,
        args.output_summary,
        commit_sha=args.commit_sha,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
    )
    return 0 if len(report["candidates"]) == DSR_TRIAL_COUNT else 1


if __name__ == "__main__":
    raise SystemExit(main())
