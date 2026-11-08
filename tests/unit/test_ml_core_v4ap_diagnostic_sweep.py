from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts import fetch_binance_public_ohlcv as fetcher
from scripts import run_ml_core_v4ap_diagnostic_sweep as sweep


def make_bars(close: np.ndarray, *, start: str = "2025-01-01") -> pd.DataFrame:
    values = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=len(values), freq="15min", tz="UTC"),
            "open": values,
            "high": values * 1.002,
            "low": values * 0.998,
            "close": values,
            "volume": np.ones(len(values), dtype=float),
        }
    )


def test_preregistered_grid_has_exactly_36_unique_cells() -> None:
    identifiers = sweep._grid_candidate_ids()
    assert sweep.DSR_TRIAL_COUNT == 36
    assert len(identifiers) == 36
    assert len(sweep.ASSETS) == 2
    assert len(sweep.TIMEFRAMES) == 3
    assert len(sweep.STRATEGIES) == 2
    assert len(sweep.EXIT_GEOMETRIES) == 3


def test_execution_cost_contract_is_exactly_30_bps_round_trip() -> None:
    execution = sweep._execution_config()
    assert execution.fee_bps == 10.0
    assert execution.slippage_bps == 5.0
    assert execution.position_fraction == 1.0
    assert sweep.ROUND_TRIP_COST_BPS == 30.0


def test_mean_reversion_requires_rsi_below_30_and_close_above_ema200() -> None:
    rising = np.linspace(80.0, 120.0, 260)
    decline = np.linspace(120.0, 115.0, 32)
    bars = make_bars(np.concatenate((rising, decline)))
    signal, diagnostics = sweep.generate_mean_reversion_signal(bars)
    assert bool(signal.iloc[-1])
    assert diagnostics["rsi_threshold"] == 30.0
    assert diagnostics["ema_period"] == 200
    assert diagnostics["lookahead_used"] is False
    assert diagnostics["hmm_used"] is False
    assert diagnostics["cusum_used"] is False


def test_mean_reversion_rejects_non_positive_or_non_finite_close() -> None:
    bars = make_bars(np.full(210, 100.0))
    bars.loc[10, "close"] = np.nan
    with pytest.raises(ValueError, match="v4ap_mean_reversion_close_invalid"):
        sweep.generate_mean_reversion_signal(bars)


def test_donchian_uses_only_previous_20_completed_highs() -> None:
    bars = make_bars(np.full(25, 100.0))
    bars.loc[20, ["close", "high"]] = [101.0, 102.0]
    bars.loc[21, ["close", "high"]] = [101.0, 105.0]
    signal, diagnostics = sweep.generate_donchian_breakout_signal(bars)
    assert bool(signal.iloc[20])
    assert not bool(signal.iloc[21])
    assert diagnostics["lookback_bars"] == 20
    assert diagnostics["lookahead_used"] is False


def test_execution_signals_are_next_bar_and_have_no_entry_at_boundary() -> None:
    bars = make_bars(np.full(8, 100.0))
    primary = pd.Series([True, False, True, True, False, True, True, True])
    actions, entries, gap_flattens = sweep._build_execution_signals(
        bars, primary, timeframe="15m", horizon_bars=2
    )
    buys = [item for item in actions if item["signal"] == "BUY"]
    terminal = [item for item in actions if item["signal"] == "NO_ENTRY"]
    assert entries == len(buys)
    assert entries == 4
    assert any(
        pd.Timestamp(item["signal_timestamp"]) == bars["date"].iloc[5]
        and pd.Timestamp(item["execution_timestamp"]) == bars["date"].iloc[6]
        for item in buys
    )
    assert gap_flattens == 0
    assert len(terminal) == 1
    assert all(
        pd.Timestamp(item["execution_timestamp"])
        - pd.Timestamp(item["signal_timestamp"])
        == pd.Timedelta(minutes=15)
        for item in buys
    )
    assert all(item["signal"] != "BUY" for item in actions[-1:])


def test_execution_signals_flatten_at_timeframe_discontinuity() -> None:
    bars = make_bars(np.full(6, 100.0))
    bars.loc[3:, "date"] = bars.loc[3:, "date"] + timedelta(minutes=15)
    primary = pd.Series([True, False, True, True, False, True])
    actions, _entries, gap_flattens = sweep._build_execution_signals(
        bars, primary, timeframe="15m", horizon_bars=1
    )
    assert gap_flattens == 1
    assert any(item["signal"] == "SELL" for item in actions)



@pytest.mark.parametrize("symbol", sweep.ASSETS)
def test_minute_cadence_accepts_only_documented_spot_gap(symbol: str) -> None:
    dates = pd.Series(
        pd.to_datetime(
            [
                "2023-03-24T12:38:00Z",
                "2023-03-24T12:39:00Z",
                "2023-03-24T14:00:00Z",
                "2023-03-24T14:01:00Z",
            ],
            utc=True,
        )
    )
    sweep._validate_minute_cadence(
        dates,
        symbol=symbol,
        missing_minutes=80,
        start_utc=pd.Timestamp("2023-03-24T12:38:00Z"),
        end_utc=pd.Timestamp("2023-03-24T14:01:00Z"),
    )

def test_minute_cadence_rejects_undocumented_gap() -> None:
    dates = pd.Series(
        pd.to_datetime(
            [
                "2023-03-24T12:38:00Z",
                "2023-03-24T12:39:00Z",
                "2023-03-24T14:01:00Z",
            ],
            utc=True,
        )
    )
    with pytest.raises(ValueError, match="v4ap_market_gap_timestamp_mismatch"):
        sweep._validate_minute_cadence(
            dates,
            symbol="ETHUSDT",
            missing_minutes=80,
            start_utc=pd.Timestamp("2023-03-24T12:38:00Z"),
            end_utc=pd.Timestamp("2023-03-24T14:01:00Z"),
        )


def test_dsr_uses_all_36_preregistered_trials() -> None:
    candidates: list[dict[str, Any]] = []
    for index in range(sweep.DSR_TRIAL_COUNT):
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.015, -0.002])
        diagnostics = sweep._trade_return_diagnostics(
            [{"net_return_fraction": float(value)} for value in returns]
        )
        candidates.append({"candidate_id": str(index), **diagnostics})
    cohort = sweep._attach_dsr_to_cohort(candidates)
    assert cohort["trial_count"] == 36
    assert cohort["valid_sharpe_estimate_count"] == 36
    assert all(item["dsr_trial_count"] == 36 for item in candidates)
    assert all(item["dsr_status"] == "COMPUTED" for item in candidates)


def test_acceptance_thresholds_are_inclusive_except_dsr_z() -> None:
    candidate = {
        "trade_count": 50,
        "win_rate": 0.40,
        "net_profit_factor": 1.10,
        "net_profit_factor_infinite_no_losses": False,
        "dsr_z_score": 0.00001,
    }
    assert sweep._candidate_qualifies(candidate)
    candidate["dsr_z_score"] = 0.0
    assert not sweep._candidate_qualifies(candidate)


def test_no_edge_summary_contains_exact_required_conclusion() -> None:
    candidates = []
    for identifier in sorted(sweep._grid_candidate_ids()):
        symbol, timeframe, strategy, geometry = identifier.split("|")
        candidates.append(
            {
                "candidate_id": identifier,
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy,
                "geometry": geometry,
                "raw_signal_count": 0,
                "trade_count": 0,
                "win_rate": 0.0,
                "net_profit_factor": None,
                "net_profit_factor_infinite_no_losses": False,
                "max_drawdown_percent": 0.0,
                "dsr_z_score": None,
                "net_pnl_usd": 0.0,
                "net_expectancy_usd_per_trade": None,
                "qualifies": False,
            }
        )
    report: dict[str, Any] = {
        "commit_sha": "a" * 40,
        "dataset": {
            "start_utc": "2021-11-02T00:00:00Z",
            "end_utc": "2026-11-01T23:59:00Z",
            "assets": {
                symbol: {
                    "row_count": 1,
                    "missing_minute_intervals": 0,
                    "csv_sha256": "b" * 64,
                }
                for symbol in sweep.ASSETS
            },
        },
        "candidates": candidates,
        "qualifying_candidate_ids": [],
        "conclusion": sweep.CONCLUSION_NO_EDGE,
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "credentials_used": False,
        },
    }
    summary = sweep.render_step_summary(report)
    assert sweep.CONCLUSION_NO_EDGE in summary
    results_section = summary.split("## Results: all 36 combinations", 1)[1].split(
        "## Conclusion", 1
    )[0]
    result_rows = [
        line
        for line in results_section.splitlines()
        if line.startswith("| ETHUSDT |") or line.startswith("| BTCUSDT |")
    ]
    assert len(result_rows) == 36


def test_manifest_rejects_private_or_unbound_data() -> None:
    start = pd.Timestamp("2021-11-02T00:00:00Z")
    end = start + pd.Timedelta(days=1825) - pd.Timedelta(minutes=1)
    manifest = {
        "schema_version": fetcher.SCHEMA_VERSION,
        "provider": "Binance Spot public klines API",
        "symbol": "BTCUSDT",
        "pair": "BTC/USDT",
        "interval": "1m",
        "requested_days": 1825,
        "start_utc": sweep._iso_utc(start),
        "end_utc": sweep._iso_utc(end),
        "row_count": 1825 * 1440,
        "missing_minute_intervals": 0,
        "missing_gap_ranges": [],
        "csv_sha256": "c" * 64,
        "commit_sha": "a" * 40,
        "public_api_only": True,
        "credentials_used": False,
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "exchange_calls": 0,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }
    assert (
        sweep._validate_manifest(
            manifest,
            symbol="BTCUSDT",
            commit_sha="a" * 40,
            start_utc=start,
            end_utc=end,
            csv_digest="c" * 64,
        )
        == 0
    )
    manifest["credentials_used"] = True
    with pytest.raises(ValueError, match="v4ap_market_manifest_contract_invalid"):
        sweep._validate_manifest(
            manifest,
            symbol="BTCUSDT",
            commit_sha="a" * 40,
            start_utc=start,
            end_utc=end,
            csv_digest="c" * 64,
        )


@pytest.mark.parametrize("symbol", sweep.ASSETS)
def test_manifest_accepts_only_documented_spot_gap_for_each_asset(
    symbol: str,
) -> None:
    start = pd.Timestamp("2021-11-02T00:00:00Z")
    end = start + pd.Timedelta(days=1825) - pd.Timedelta(minutes=1)
    manifest = {
        "schema_version": fetcher.SCHEMA_VERSION,
        "provider": "Binance Spot public klines API",
        "symbol": symbol,
        "pair": f"{symbol[:-4]}/USDT",
        "interval": "1m",
        "requested_days": 1825,
        "start_utc": sweep._iso_utc(start),
        "end_utc": sweep._iso_utc(end),
        "row_count": 1825 * 1440 - 80,
        "missing_minute_intervals": 80,
        "missing_gap_ranges": [
            {
                "start_utc": "2023-03-24T12:40:00Z",
                "end_utc": "2023-03-24T13:59:00Z",
                "missing_minute_intervals": 80,
            }
        ],
        "csv_sha256": "d" * 64,
        "commit_sha": "a" * 40,
        "public_api_only": True,
        "credentials_used": False,
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "exchange_calls": 0,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }
    assert (
        sweep._validate_manifest(
            manifest,
            symbol=symbol,
            commit_sha="a" * 40,
            start_utc=start,
            end_utc=end,
            csv_digest="d" * 64,
        )
        == 80
    )


def test_request_requires_exact_1825_day_minute_aligned_window() -> None:
    start = "2021-11-02T00:00:00Z"
    end = "2026-10-31T23:59:00Z"
    parsed_start, parsed_end = sweep._validate_request("a" * 40, start, end)
    assert parsed_end - parsed_start == pd.Timedelta(days=1825) - pd.Timedelta(
        minutes=1
    )
    with pytest.raises(
        ValueError, match="v4ap_source_window_must_cover_exact_1825_days"
    ):
        sweep._validate_request("a" * 40, start, "2026-10-31T23:58:00Z")
