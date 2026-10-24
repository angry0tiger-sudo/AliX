"""Proxy-only Stage B liquidity sensitivity. Never authorizes economic acceptance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from pandas import DataFrame

from scripts import run_stage_b_triple_barrier_orderbook as stage_b
from services.meta_labeling.triple_barrier import (
    TripleBarrierConfig,
    apply_triple_barrier,
)

SCHEMA = "alix.stage-b.proxy-liquidity-sensitivity.v1"
SYMBOL = "BTCUSDT"
BINANCE_KLINE_INTERVAL = "1h"
BINANCE_SPOT_ROOT = "https://data.binance.vision/data/spot"
PROXY_DAILY_QUOTE_CAP = 1_500_000.0
MAX_BINANCE_VOLUME_SHARE = 0.10
PARTICIPATION_CAP = 0.01
REFERENCE_ORDER_NOTIONAL = 10.0
HORIZONS = (24, 48)
PT_MULTIPLIERS = (2.0, 3.0)
SL_MULTIPLIERS = (0.75, 1.25)
TARGET_VOLATILITY_FLOOR = 0.0051
BOOK_TIME = "timestamp"
BOOK_BANDS = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
BOOK_FEATURE_BANDS = (1, 2, 5)
BOOK_MAX_AGE_SECONDS = 60
PRICE_BAND_TOLERANCE_FRACTION = 0.005
BOOK_MANIFEST_SCHEMA = "alix.binance-bookdepth-manifest.v1"


def _download_bytes(url: str) -> bytes | None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "AliX-research-only/1.0"}, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"binance_archive_http_status:{response.status}:{url}"
                )
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise RuntimeError(f"binance_archive_http_error:{error.code}:{url}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"binance_archive_download_error:{url}:{error.reason}"
        ) from error


def _month_url(month: pd.Timestamp) -> str:
    return (
        f"{BINANCE_SPOT_ROOT}/monthly/klines/{SYMBOL}/{BINANCE_KLINE_INTERVAL}/"
        f"{SYMBOL}-{BINANCE_KLINE_INTERVAL}-{month:%Y-%m}.zip"
    )


def _day_url(day: pd.Timestamp) -> str:
    return (
        f"{BINANCE_SPOT_ROOT}/daily/klines/{SYMBOL}/{BINANCE_KLINE_INTERVAL}/"
        f"{SYMBOL}-{BINANCE_KLINE_INTERVAL}-{day:%Y-%m-%d}.zip"
    )


def _read_kline_archive(payload: bytes) -> DataFrame:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv_names = [
                name for name in archive.namelist() if name.lower().endswith(".csv")
            ]
            if len(csv_names) != 1:
                raise ValueError("binance_kline_archive_csv_count_invalid")
            with archive.open(csv_names[0]) as stream:
                frame = pd.read_csv(stream, header=None, usecols=[0, 1, 2, 3, 4, 5, 7])
    except (zipfile.BadZipFile, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"binance_kline_archive_invalid:{error}") from error

    frame.columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
    ]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open_time", "close", "quote_volume"]).copy()
    if frame.empty:
        raise ValueError("binance_kline_archive_has_no_numeric_rows")
    if (frame["close"] <= 0).any() or (frame["quote_volume"] < 0).any():
        raise ValueError("binance_kline_price_or_quote_volume_out_of_range")

    unit = "us" if float(frame["open_time"].max()) >= 100_000_000_000_000 else "ms"
    frame["date"] = pd.to_datetime(
        frame["open_time"].astype("int64"), unit=unit, utc=True, errors="raise"
    )
    result = frame.loc[:, ["date", "close", "quote_volume"]].sort_values("date")
    if result["date"].duplicated().any():
        raise ValueError("binance_kline_archive_duplicate_timestamps")
    return result.reset_index(drop=True)


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    cursor = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    last = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    months: list[pd.Timestamp] = []
    while cursor <= last:
        months.append(cursor)
        cursor = cursor + pd.offsets.MonthBegin(1)
    return months


def _download_binance_klines(
    start: pd.Timestamp, end: pd.Timestamp
) -> tuple[DataFrame, dict[str, Any]]:
    pieces: list[DataFrame] = []
    downloads: list[dict[str, Any]] = []
    missing_archives: list[str] = []
    start_month = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_month = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")

    for month in _month_starts(start, end):
        url = _month_url(month)
        payload = _download_bytes(url)
        if payload is not None:
            pieces.append(_read_kline_archive(payload))
            downloads.append(
                {
                    "url": url,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "kind": "monthly",
                }
            )
            continue

        # Incomplete boundary months may lack a monthly archive. Only boundary months
        # fall back to official daily archives; missing interior months remain missing.
        if month not in (start_month, end_month):
            missing_archives.append(url)
            continue

        first_day = max(month, start.floor("D"))
        last_day = min(month + pd.offsets.MonthEnd(1), end.floor("D"))
        day = first_day
        while day <= last_day:
            daily_url = _day_url(day)
            daily_payload = _download_bytes(daily_url)
            if daily_payload is None:
                missing_archives.append(daily_url)
            else:
                pieces.append(_read_kline_archive(daily_payload))
                downloads.append(
                    {
                        "url": daily_url,
                        "sha256": hashlib.sha256(daily_payload).hexdigest(),
                        "kind": "daily",
                    }
                )
            day = day + pd.Timedelta(days=1)

    if not pieces:
        raise ValueError("binance_kline_source_returned_no_data")
    klines = pd.concat(pieces, ignore_index=True)
    klines = (
        klines.loc[
            klines["date"].between(start.floor("D"), end + pd.Timedelta(hours=1))
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if klines.empty:
        raise ValueError("binance_kline_source_has_no_rows_in_requested_period")
    if klines["date"].duplicated().any():
        raise ValueError("binance_kline_source_ambiguous_duplicate_timestamps")
    if not klines["date"].is_monotonic_increasing:
        raise ValueError("binance_kline_source_timestamps_not_monotonic")

    manifest = {
        "schema": "alix.binance-spot-kline-source-manifest.v1",
        "venue": "binance_spot",
        "symbol": SYMBOL,
        "interval": BINANCE_KLINE_INTERVAL,
        "timestamp_semantics": "UTC candle-open timestamps",
        "volume_semantics": "Binance source quote_asset_volume; never treated as Nobitex observed volume",
        "archive_count": len(downloads),
        "downloaded_archives": downloads,
        "missing_archives": missing_archives,
        "rows": len(klines),
        "first_timestamp_utc": klines["date"].iloc[0].isoformat(),
        "last_timestamp_utc": klines["date"].iloc[-1].isoformat(),
    }
    return klines, manifest


def _scale_binance_quote_volume(klines: DataFrame) -> tuple[DataFrame, dict[str, Any]]:
    if not {"date", "quote_volume"}.issubset(klines.columns) or klines.empty:
        raise ValueError("binance_quote_volume_schema_or_data_invalid")
    result = klines.loc[:, ["date", "quote_volume"]].copy()
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
    result["quote_volume"] = pd.to_numeric(result["quote_volume"], errors="raise")
    values = result["quote_volume"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("binance_quote_volume_non_finite_or_negative")
    if result["date"].duplicated().any():
        raise ValueError("binance_quote_volume_timestamp_duplicate")

    result["utc_day"] = result["date"].dt.floor("D")
    daily_total = (
        result.groupby("utc_day")["quote_volume"].transform("sum").to_numpy(float)
    )
    ratio = np.zeros(len(result), dtype=float)
    positive = daily_total > 0
    ratio[positive] = np.minimum(
        MAX_BINANCE_VOLUME_SHARE, PROXY_DAILY_QUOTE_CAP / daily_total[positive]
    )
    result["proxy_scale_ratio"] = ratio
    result["proxy_quote_volume"] = values * ratio
    daily_proxy = result.groupby("utc_day")["proxy_quote_volume"].sum()
    if (daily_proxy > PROXY_DAILY_QUOTE_CAP + 1e-6).any():
        raise ValueError("proxy_daily_quote_cap_invariant_violated")
    if (ratio > MAX_BINANCE_VOLUME_SHARE + 1e-12).any():
        raise ValueError("proxy_binance_discount_invariant_violated")

    stats = {
        "rule": "per UTC day, scale actual Binance hourly quote turnover by min(10%, 1,500,000 / Binance daily quote turnover)",
        "daily_quote_cap": PROXY_DAILY_QUOTE_CAP,
        "maximum_binance_share": MAX_BINANCE_VOLUME_SHARE,
        "minimum_discount_from_binance": 0.90,
        "max_applied_scale_ratio": float(ratio.max()),
        "days_with_positive_binance_quote_volume": int((daily_proxy > 0).sum()),
        "max_daily_proxy_quote_volume": float(daily_proxy.max()),
        "historical_nobitex_volume_replaced": False,
        "proxy_is_not_observed_nobitex_volume": True,
    }
    return result, stats


def _align_proxy_volume(
    candles: DataFrame, klines: DataFrame
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align proxy labels to the fixed candle phase without claiming identical windows."""
    scaled, scale_stats = _scale_binance_quote_volume(klines)
    candle_dates = pd.to_datetime(candles["date"], utc=True, errors="raise")
    minute_phases = sorted(candle_dates.dt.minute.unique().tolist())
    if (
        len(minute_phases) != 1
        or candle_dates.dt.second.ne(0).any()
        or candle_dates.dt.microsecond.ne(0).any()
        or candle_dates.dt.nanosecond.ne(0).any()
    ):
        raise ValueError("proxy_candle_timestamp_phase_not_uniform")
    if (
        scaled["date"].dt.minute.ne(0).any()
        or scaled["date"].dt.second.ne(0).any()
        or scaled["date"].dt.microsecond.ne(0).any()
        or scaled["date"].dt.nanosecond.ne(0).any()
    ):
        raise ValueError("binance_hourly_source_timestamps_not_hour_aligned")

    # Nobitex candles in the pinned source are timestamped at HH:30 UTC,
    # while Binance 1h kline labels are HH:00 UTC. Shift the label only by
    # the constant phase so observations align for sensitivity analysis. The
    # volume still represents Binance's original [HH:00, HH+1:00) interval,
    # not the exact Nobitex [HH:30, HH+1:30) interval.
    phase_minutes = int(minute_phases[0])
    aligned_source = scaled.loc[
        :, ["date", "quote_volume", "proxy_quote_volume", "proxy_scale_ratio"]
    ].copy()
    aligned_source["date"] = aligned_source["date"] + pd.Timedelta(
        minutes=phase_minutes
    )

    aligned = candles.loc[:, ["date"]].copy()
    aligned["date"] = candle_dates
    aligned = aligned.merge(
        aligned_source,
        how="left",
        on="date",
        validate="one_to_one",
        indicator=True,
    )
    matched = aligned["_merge"].eq("both")
    proxy = aligned["proxy_quote_volume"].fillna(0.0).to_numpy(dtype=float)
    source_quote = aligned["quote_volume"]
    capacity = proxy * PARTICIPATION_CAP
    if not np.isfinite(proxy).all() or np.any(proxy < 0):
        raise ValueError("proxy_volume_alignment_invalid")
    if not np.isfinite(capacity).all() or np.any(capacity < 0):
        raise ValueError("proxy_capacity_invalid")

    daily_proxy = (
        pd.DataFrame({"date": aligned["date"], "proxy_quote_volume": proxy})
        .groupby(aligned["date"].dt.floor("D"))["proxy_quote_volume"]
        .sum()
    )
    if (daily_proxy > PROXY_DAILY_QUOTE_CAP + 1e-6).any():
        raise ValueError("aligned_proxy_daily_quote_cap_invariant_violated")

    stats = {
        **scale_stats,
        "source": "Binance Spot BTCUSDT hourly quote_asset_volume archives",
        "candle_count": len(candles),
        "matched_candle_count": int(matched.sum()),
        "unmatched_candle_count": int((~matched).sum()),
        "matched_candle_fraction": float(matched.mean()) if len(matched) else 0.0,
        "matched_rows_with_positive_binance_quote_volume": int(
            (matched & source_quote.fillna(0).gt(0)).sum()
        ),
        "proxy_candle_rows_with_10_quote_unit_capacity": int(
            (capacity >= REFERENCE_ORDER_NOTIONAL).sum()
        ),
        "proxy_candle_rows_with_positive_capacity": int((capacity > 0).sum()),
        "proxy_daily_max_over_aligned_candles": float(daily_proxy.max()),
        "participation_cap_fraction": PARTICIPATION_CAP,
        "order_notional_quote_currency": REFERENCE_ORDER_NOTIONAL,
        "required_proxy_candle_quote_turnover_for_10_unit_order": (
            REFERENCE_ORDER_NOTIONAL / PARTICIPATION_CAP
        ),
        "unmatched_rows_receive_zero_capacity": True,
        "timestamp_phase_minutes": phase_minutes,
        "timestamp_alignment_policy": (
            "label-only shift by the uniform Nobitex timestamp phase; Binance volume "
            "retains its original UTC-hour interval and is only a nearby-window sensitivity proxy"
        ),
        "exact_same_window_turnover_claimed": False,
    }
    return capacity, stats


def _event_outcomes(
    candles: DataFrame,
) -> tuple[dict[tuple[int, float, float], DataFrame], list[dict[str, Any]]]:
    close = pd.to_numeric(candles["close"], errors="raise")
    volatility = np.log(close).diff().rolling(24, min_periods=24).std() * math.sqrt(24)
    outcome_map: dict[tuple[int, float, float], DataFrame] = {}
    diagnostics: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        event_indices = [
            index
            for index in range(24, len(candles) - horizon, horizon)
            if math.isfinite(float(volatility.iloc[index]))
        ]
        if not event_indices:
            raise ValueError(f"triple_barrier_event_grid_empty:{horizon}")
        targets = [
            max(float(volatility.iloc[index]), TARGET_VOLATILITY_FLOOR)
            for index in event_indices
        ]
        events = DataFrame(
            {
                "event_index": event_indices,
                "side": [1] * len(event_indices),
                "target_volatility": targets,
            }
        )
        for pt in PT_MULTIPLIERS:
            for sl in SL_MULTIPLIERS:
                config = TripleBarrierConfig(
                    profit_taking_multiplier=pt,
                    stop_loss_multiplier=sl,
                    max_holding_bars=horizon,
                    fee_bps_per_side=0.0,
                    slippage_bps_per_side=0.0,
                    timeframe_seconds=stage_b.HOUR,
                )
                outcomes = apply_triple_barrier(candles, events, config)
                outcome_map[(horizon, pt, sl)] = outcomes
                gross = outcomes["gross_return"].to_numpy(dtype=float)
                for slip in stage_b.SLIPPAGE_BPS:
                    cost = 2.0 * (stage_b.FEE_BPS_SIDE + slip) / 10_000.0
                    diagnostics.append(
                        {
                            "horizon_hours": horizon,
                            "profit_taking_multiplier": pt,
                            "stop_loss_multiplier": sl,
                            "target_volatility_floor": TARGET_VOLATILITY_FLOOR,
                            "minimum_take_profit_barrier_return": pt
                            * TARGET_VOLATILITY_FLOOR,
                            "slippage_bps_per_side": slip,
                            "round_trip_cost_bps": 2.0 * (stage_b.FEE_BPS_SIDE + slip),
                            "scheduled_event_count_not_trade_count": len(outcomes),
                            "take_profit_count": int(
                                outcomes["exit_reason"].eq("take_profit").sum()
                            ),
                            "stop_loss_count": int(
                                outcomes["exit_reason"].eq("stop_loss").sum()
                            ),
                            "vertical_count": int(
                                outcomes["exit_reason"].eq("vertical").sum()
                            ),
                            "mean_gross_barrier_return": float(np.mean(gross)),
                            "median_gross_barrier_return": float(np.median(gross)),
                            "mean_net_barrier_diagnostic_return": float(
                                np.mean(gross - cost)
                            ),
                            "status": "DESCRIPTIVE_BARRIER_OUTCOMES_NOT_EXECUTION_BACKTEST",
                        }
                    )
    return outcome_map, diagnostics


def _validate_reference_prices(reference: DataFrame) -> DataFrame:
    if (
        not {"timestamp", "reference_price"}.issubset(reference.columns)
        or reference.empty
    ):
        raise ValueError("bookdepth_reference_price_schema_missing_or_empty")
    result = reference.loc[:, ["timestamp", "reference_price"]].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    result["reference_price"] = pd.to_numeric(
        result["reference_price"], errors="coerce"
    )
    values = result["reference_price"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("bookdepth_reference_price_invalid")
    if result["timestamp"].duplicated().any():
        raise ValueError("bookdepth_reference_timestamps_duplicate")
    if not result["timestamp"].is_monotonic_increasing:
        raise ValueError("bookdepth_reference_timestamps_not_monotonic")
    return result.rename(columns={"timestamp": "reference_timestamp"})


def _validate_bookdepth_rows(book: DataFrame) -> DataFrame:
    required = {BOOK_TIME, "percentage", "depth", "notional"}
    if not required.issubset(book.columns) or book.empty:
        raise ValueError("bookdepth_schema_missing_or_empty")
    result = book.loc[:, [BOOK_TIME, "percentage", "depth", "notional"]].copy()
    result[BOOK_TIME] = pd.to_datetime(result[BOOK_TIME], utc=True, errors="raise")
    for column in ("percentage", "depth", "notional"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    values = result[["percentage", "depth", "notional"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("bookdepth_non_finite_values")
    if not result["percentage"].isin(BOOK_BANDS).all():
        raise ValueError("bookdepth_percentage_band_unsupported")
    if (result["depth"] <= 0).any() or (result["notional"] <= 0).any():
        raise ValueError("bookdepth_depth_or_notional_not_positive")
    if result.duplicated([BOOK_TIME, "percentage"]).any():
        raise ValueError("bookdepth_timestamp_percentage_duplicate")

    expected = set(BOOK_BANDS)
    for _, group in result.groupby(BOOK_TIME, sort=True):
        if set(group["percentage"].astype(int).tolist()) != expected or len(
            group
        ) != len(expected):
            raise ValueError("bookdepth_snapshot_incomplete_percentage_bands")
        for side in ("bid", "ask"):
            side_values = group.loc[
                group["percentage"].lt(0)
                if side == "bid"
                else group["percentage"].gt(0)
            ].copy()
            side_values["band"] = side_values["percentage"].abs().astype(int)
            side_values = side_values.sort_values("band")
            for column in ("depth", "notional"):
                if np.any(np.diff(side_values[column].to_numpy(dtype=float)) < 0):
                    raise ValueError(
                        f"bookdepth_cumulative_{column}_not_monotonic:{side}"
                    )
            implied_prices = side_values["notional"].to_numpy(
                dtype=float
            ) / side_values["depth"].to_numpy(dtype=float)
            if side == "bid" and np.any(np.diff(implied_prices) > 1e-9):
                raise ValueError("bookdepth_bid_implied_price_not_monotonic")
            if side == "ask" and np.any(np.diff(implied_prices) < -1e-9):
                raise ValueError("bookdepth_ask_implied_price_not_monotonic")
    return result.sort_values([BOOK_TIME, "percentage"]).reset_index(drop=True)


def add_banded_orderbook_features(
    candles: DataFrame, book: DataFrame, reference_prices: DataFrame
) -> DataFrame:
    """Adapt percentage-band depth to normalized features only; never infer L2 levels."""
    bands = _validate_bookdepth_rows(book)
    references = _validate_reference_prices(reference_prices)
    enriched = pd.merge_asof(
        bands.sort_values(BOOK_TIME),
        references.sort_values("reference_timestamp"),
        left_on=BOOK_TIME,
        right_on="reference_timestamp",
        direction="backward",
        tolerance=pd.Timedelta(seconds=BOOK_MAX_AGE_SECONDS),
        allow_exact_matches=True,
    )
    if enriched["reference_price"].isna().any():
        raise ValueError("bookdepth_reference_price_missing_or_stale")

    implied = enriched["notional"].to_numpy(dtype=float) / enriched["depth"].to_numpy(
        dtype=float
    )
    reference = enriched["reference_price"].to_numpy(dtype=float)
    pct = enriched["percentage"].to_numpy(dtype=float) / 100.0
    tolerance = reference * PRICE_BAND_TOLERANCE_FRACTION
    lower = np.where(pct < 0, reference * (1.0 + pct), reference)
    upper = np.where(pct > 0, reference * (1.0 + pct), reference)
    if np.any(implied < lower - tolerance) or np.any(implied > upper + tolerance):
        raise ValueError("bookdepth_implied_price_outside_declared_band")

    snapshots: list[dict[str, Any]] = []
    for timestamp, group in enriched.groupby(BOOK_TIME, sort=True):
        row: dict[str, Any] = {BOOK_TIME: timestamp}
        lookup = {
            int(record["percentage"]): record for record in group.to_dict("records")
        }
        for band in BOOK_FEATURE_BANDS:
            bid = lookup[-band]
            ask = lookup[band]
            row[f"depth_imbalance_{band}pct"] = (
                float(bid["depth"]) - float(ask["depth"])
            ) / (float(bid["depth"]) + float(ask["depth"]))
            row[f"notional_imbalance_{band}pct"] = (
                float(bid["notional"]) - float(ask["notional"])
            ) / (float(bid["notional"]) + float(ask["notional"]))
        snapshots.append(row)

    features = DataFrame(snapshots)
    left = candles.loc[:, ["date"]].copy()
    left["date"] = pd.to_datetime(left["date"], utc=True, errors="raise")
    left["decision_time"] = left["date"] + pd.Timedelta(seconds=stage_b.HOUR)
    joined = pd.merge_asof(
        left.sort_values("decision_time"),
        features.sort_values(BOOK_TIME),
        left_on="decision_time",
        right_on=BOOK_TIME,
        direction="backward",
        tolerance=pd.Timedelta(seconds=BOOK_MAX_AGE_SECONDS),
        allow_exact_matches=True,
    )
    feature_columns = [
        f"{kind}_imbalance_{band}pct"
        for band in BOOK_FEATURE_BANDS
        for kind in ("depth", "notional")
    ]
    if joined[BOOK_TIME].isna().any():
        raise ValueError("bookdepth_snapshot_missing_or_stale_at_decision_time")
    values = joined[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1.0 + 1e-12):
        raise ValueError("bookdepth_relative_feature_out_of_range")
    return joined.loc[:, ["date", *feature_columns]].reset_index(drop=True)


def _validate_bookdepth_manifest(
    book_path: Path, reference_path: Path, manifest_path: Path
) -> dict[str, Any]:
    if (
        not book_path.is_file()
        or not reference_path.is_file()
        or not manifest_path.is_file()
    ):
        raise ValueError("bookdepth_data_reference_or_manifest_missing")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("bookdepth_manifest_not_object")
    required = {
        "schema",
        "venue",
        "symbol",
        "source_url",
        "permission_status",
        "csv_sha256",
        "reference_price_sha256",
    }
    if not required.issubset(value):
        raise ValueError("bookdepth_manifest_required_fields_missing")
    if value["schema"] != BOOK_MANIFEST_SCHEMA:
        raise ValueError("bookdepth_manifest_schema_unsupported")
    if value["venue"] != "binance_usdm_futures" or value["symbol"] != SYMBOL:
        raise ValueError("bookdepth_manifest_identity_mismatch")
    if value["permission_status"] != "approved":
        raise ValueError("bookdepth_permission_not_approved")
    source = urlsplit(str(value["source_url"]))
    if (
        source.scheme != "https"
        or source.hostname != "data.binance.vision"
        or "/data/futures/um/" not in source.path
        or "/bookDepth/BTCUSDT/" not in source.path
    ):
        raise ValueError("bookdepth_manifest_source_url_invalid")
    book_sha = hashlib.sha256(book_path.read_bytes()).hexdigest()
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    if book_sha != str(value["csv_sha256"]) or len(book_sha) != 64:
        raise ValueError("bookdepth_csv_checksum_mismatch")
    if (
        reference_sha != str(value["reference_price_sha256"])
        or len(reference_sha) != 64
    ):
        raise ValueError("bookdepth_reference_price_checksum_mismatch")
    return value


def _load_banded_book_features(
    candles: DataFrame,
    book_path: Path | None,
    reference_path: Path | None,
    manifest_path: Path | None,
) -> tuple[str, DataFrame | None, int | None, str | None, list[str]]:
    if book_path is None and reference_path is None and manifest_path is None:
        return (
            "NOT_PROVIDED",
            None,
            None,
            None,
            ["historical_banded_orderbook_not_provided"],
        )
    if book_path is None or reference_path is None or manifest_path is None:
        return (
            "INVALID",
            None,
            None,
            None,
            ["bookdepth_csv_reference_price_and_manifest_are_all_required"],
        )
    try:
        _validate_bookdepth_manifest(book_path, reference_path, manifest_path)
        book = pd.read_csv(book_path)
        references = pd.read_csv(reference_path)
        features = add_banded_orderbook_features(candles, book, references)
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ) as error:
        return (
            "INVALID_OR_INCOMPLETE",
            None,
            None,
            None,
            [f"bookdepth_validation_failed:{error}"],
        )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return "VALIDATED_BANDED_RELATIVE_FEATURES", features, len(book), manifest_sha, []


def _proxy_passing_candidates(stress_rows: list[dict[str, Any]]) -> list[str]:
    grouped: dict[str, list[bool]] = {}
    for row in stress_rows:
        candidate = str(row["candidate_id"])
        grouped.setdefault(candidate, []).append(
            bool(row["strict_economic_gate_on_executable_trades"]["passed"])
        )
    return sorted(
        candidate
        for candidate, results in grouped.items()
        if len(results) == len(stage_b.SLIPPAGE_BPS) and all(results)
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status", "reason"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    input_dir: Path,
    output_dir: Path,
    commit_sha: str,
    bookdepth_csv: Path | None = None,
    reference_price_csv: Path | None = None,
    bookdepth_manifest: Path | None = None,
) -> dict[str, Any]:
    if len(commit_sha) != 40 or any(
        char not in "0123456789abcdef" for char in commit_sha
    ):
        raise ValueError("research_commit_sha_invalid")
    candles, source_report, segment_hash = stage_b._load_development(input_dir)
    klines, kline_manifest = _download_binance_klines(
        pd.Timestamp(candles["date"].iloc[0]), pd.Timestamp(candles["date"].iloc[-1])
    )
    capacity, proxy_stats = _align_proxy_volume(candles, klines)
    outcomes_by_config, barrier_diagnostics = _event_outcomes(candles)
    orderbook_status, book_features, book_rows, book_manifest_hash, book_blockers = (
        _load_banded_book_features(
            candles, bookdepth_csv, reference_price_csv, bookdepth_manifest
        )
    )
    price_features = stage_b._price_features(candles)
    feature_groups = stage_b._build_feature_groups(price_features, book_features)

    stage_b.REFERENCE_ORDER_NOTIONAL_QUOTE = REFERENCE_ORDER_NOTIONAL
    fold_rows, prediction_rows, stress_rows = stage_b._evaluate_grid(
        candles, outcomes_by_config, feature_groups, capacity
    )
    proxy_passing = _proxy_passing_candidates(stress_rows)
    run_status = (
        "COMPLETED_PROXY_SENSITIVITY"
        if float(proxy_stats["matched_candle_fraction"]) >= 0.95
        else "BLOCKED_PROXY_SOURCE_COVERAGE"
    )
    report: dict[str, Any] = {
        "report_schema": SCHEMA,
        "run_status": run_status,
        "created_by_commit_sha": commit_sha,
        "source": {
            "coverage_workflow_run_id": stage_b.SOURCE_RUN_ID,
            "coverage_csv_sha256": source_report["csv_sha256"],
            "coverage_manifest_sha256": source_report["manifest_sha256"],
            "segment_a_rows_sha256": segment_hash,
            "symbol": SYMBOL,
            "ohlcv_venue": "Nobitex historical prices; original volume field remains unchanged",
            "timeframe": "1h",
            "development_rows": len(candles),
            "reserved_final_90_day_holdout_rows": stage_b.A_HOLDOUT,
            "holdout_used": False,
            "development_start_utc": stage_b.utc_text(candles["date"].iloc[0]),
            "development_end_utc": stage_b.utc_text(candles["date"].iloc[-1]),
        },
        "protocol": {
            "horizons_hours": list(HORIZONS),
            "profit_taking_multipliers": list(PT_MULTIPLIERS),
            "stop_loss_multipliers": list(SL_MULTIPLIERS),
            "target_volatility_source": "trailing 24-hour volatility, floored at 0.51% to keep the 2x TP barrier at or above 1.02%",
            "target_volatility_floor": TARGET_VOLATILITY_FLOOR,
            "minimum_take_profit_barrier_return": min(PT_MULTIPLIERS)
            * TARGET_VOLATILITY_FLOOR,
            "minimum_take_profit_barrier_exceeds_one_percent": (
                min(PT_MULTIPLIERS) * TARGET_VOLATILITY_FLOOR > 0.01
            ),
            "barrier_candidate_count": len(outcomes_by_config),
            "model_candidate_count_planned": 2 * len(outcomes_by_config),
            "model_candidate_count_executed": len(feature_groups)
            * len(outcomes_by_config),
            "model_candidate_count_blocked_for_missing_or_invalid_book": (
                0 if book_features is not None else len(outcomes_by_config)
            ),
            "diagnostic_scenario_count": len(barrier_diagnostics),
            "oos_prediction_row_count": len(prediction_rows),
            "fee_bps_per_side": stage_b.FEE_BPS_SIDE,
            "slippage_bps_per_side": list(stage_b.SLIPPAGE_BPS),
            "round_trip_total_cost_bps": [
                2.0 * (stage_b.FEE_BPS_SIDE + slip) for slip in stage_b.SLIPPAGE_BPS
            ],
            "stress_acceptance_requires_all_slippage_scenarios": True,
            "probability_threshold_fixed_not_tuned": stage_b.PROBABILITY_THRESHOLD,
            "reference_order_notional_quote_currency": REFERENCE_ORDER_NOTIONAL,
            "liquidity_participation_cap_fraction": PARTICIPATION_CAP,
            "proxy_volume_used_as_model_feature": False,
            "proxy_used_only_for_capacity_sensitivity": True,
            "training_label_purge_rule": "training label_end_index < test_start; test label_end_index < test_stop",
        },
        "data_availability": {
            "binance_kline_proxy": proxy_stats,
            "binance_kline_archive_manifest": "binance-kline-source-manifest.json",
            "orderbook_status": orderbook_status,
            "orderbook_csv_rows": book_rows,
            "orderbook_manifest_sha256": book_manifest_hash,
            "orderbook_blockers": book_blockers,
            "bookdepth_adapter_semantics": (
                "Normalized depth/notional imbalance at 1%, 2%, and 5% bands only. "
                "No synthetic L2 levels, absolute Binance depth feature, or Nobitex liquidity inference."
            ),
        },
        "barrier_outcome_diagnostics_not_execution_backtest": barrier_diagnostics,
        "walk_forward": {
            "feature_groups": list(feature_groups),
            "fold_metrics": fold_rows,
            "candidate_cost_stress_metrics": stress_rows,
            "counterfactual_predictions_are_not_economic_evidence": True,
            "proxy_executable_trades_are_scenario_outputs_not_observed_fills": True,
        },
        "proxy_sensitivity_evaluation": {
            "status": "COMPUTED_PROXY_ONLY",
            "passing_candidates_all_three_cost_scenarios": proxy_passing,
            "proxy_candidate_pass_count": len(proxy_passing),
            "strict_gate": {
                "minimum_proxy_executable_oos_trades": 100,
                "minimum_proxy_win_rate_exclusive": 0.60,
                "minimum_net_expectancy_exclusive": 0.0,
                "maximum_absolute_drawdown_exclusive": 0.15,
                "all_three_slippage_scenarios_required": True,
                "proxy_pass_does_not_imply_economic_acceptance": True,
            },
        },
        "economic_evaluation": {
            "status": "NOT_EVALUATED_PROXY_ONLY",
            "economic_acceptance": "NO-GO",
            "executable_trade_count_is_not_observed_trade_count": True,
            "reason": (
                "Proxy turnover is inferred by discounting Binance spot quote volume and "
                "capping each UTC day at the owner-supplied BTC estimate. It is not historical "
                "Nobitex candle turnover or proof of order execution."
            ),
            "required_oos_trade_count": ">=100 executable trades",
            "required_win_rate": ">60%",
            "required_net_expectancy": ">0 after costs",
            "required_absolute_max_drawdown": "<15%",
            "all_gates_must_pass": True,
        },
        "cross_venue_limitations": [
            "BTC's owner-supplied 1,500,000 quote-unit daily estimate is a cap, not an observed value for each historical date.",
            "Hourly proxy volume is proportional to actual Binance Spot BTCUSDT quote volume, scaled by at most 10% and capped at 1,500,000 quote units per UTC day.",
            "The proxy does not modify or overwrite the zero-valued original Nobitex volume column.",
            "An order of 10 quote units passes the 1% proxy cap only when the proxy turnover of that candle is at least 1,000 quote units.",
            "The ETH and ZEC owner estimates are not applied because this Stage A artifact contains BTCUSDT only.",
            "Banded Binance bookDepth is not equivalent to a 20-level L2 snapshot and cannot supply absolute Nobitex liquidity.",
            "Banded book features remain blocked unless complete bands, independent timestamped reference prices, manifest provenance, checksum, permissions, and point-in-time coverage all pass.",
            "No data is interpolated or fabricated to fill missing volume; unmatched Binance candle timestamps receive zero capacity.",
        ],
        "safety_invariants": dict(stage_b.SAFETY),
    }
    if report["safety_invariants"] != stage_b.SAFETY:
        raise ValueError("research_safety_invariant_violation")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "proxy-liquidity-sensitivity-report.json", report)
    _write_json(output_dir / "binance-kline-source-manifest.json", kline_manifest)
    _write_csv(output_dir / "barrier-outcome-diagnostics.csv", barrier_diagnostics)
    _write_csv(output_dir / "oos-predictions.csv", prediction_rows)
    _write_csv(output_dir / "oos-candidate-cost-stress.csv", stress_rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--bookdepth-csv", type=Path, default=None)
    parser.add_argument("--reference-price-csv", type=Path, default=None)
    parser.add_argument("--bookdepth-manifest", type=Path, default=None)
    args = parser.parse_args()
    report = run_experiment(
        args.input_dir,
        args.output_dir,
        args.commit_sha,
        args.bookdepth_csv,
        args.reference_price_csv,
        args.bookdepth_manifest,
    )
    print(
        json.dumps(
            {
                "commit_sha": report["created_by_commit_sha"],
                "run_status": report["run_status"],
                "barrier_scenarios": report["protocol"]["diagnostic_scenario_count"],
                "planned_model_candidates": report["protocol"][
                    "model_candidate_count_planned"
                ],
                "executed_model_candidates": report["protocol"][
                    "model_candidate_count_executed"
                ],
                "blocked_book_candidates": report["protocol"][
                    "model_candidate_count_blocked_for_missing_or_invalid_book"
                ],
                "oos_predictions": report["protocol"]["oos_prediction_row_count"],
                "proxy_coverage_fraction": report["data_availability"][
                    "binance_kline_proxy"
                ]["matched_candle_fraction"],
                "proxy_order_capacity_rows": report["data_availability"][
                    "binance_kline_proxy"
                ]["proxy_candle_rows_with_10_quote_unit_capacity"],
                "proxy_pass_count": report["proxy_sensitivity_evaluation"][
                    "proxy_candidate_pass_count"
                ],
                "economic_status": report["economic_evaluation"]["status"],
                "Champion": report["safety_invariants"]["Champion"],
                "order_permission": report["safety_invariants"]["order_permission"],
                "trading_authorization": report["safety_invariants"][
                    "trading_authorization"
                ],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
