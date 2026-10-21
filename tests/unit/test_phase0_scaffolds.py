from typing import Any

import pandas as pd

from connectivity.manager.manager import ConnectivityManager
from services.data_quality.gate import validate_candles
from services.reporting.engine import DailyReport, ReportingEngine
from user_data.strategies.AliXStrategy import AliXStrategy


def candle(ts: int = 1000) -> dict[str, object]:
    return {
        "date": ts,
        "open": 10.0,
        "high": 12.0,
        "low": 9.0,
        "close": 11.0,
        "volume": 5.0,
    }


def test_data_quality_accepts_valid_candles() -> None:
    assert validate_candles([candle(1000), candle(1060)]).passed


def test_data_quality_fails_closed_for_empty_data() -> None:
    result = validate_candles([])
    assert not result.passed and "empty_dataset" in result.reasons


def test_data_quality_rejects_bad_ohlc_and_order() -> None:
    bad = candle(1000)
    bad["high"] = 8
    result = validate_candles([candle(1000), bad])
    assert not result.passed
    assert any("invalid_ohlc_range" in reason for reason in result.reasons)
    assert any(
        "timestamp_not_strictly_increasing" in reason for reason in result.reasons
    )


def test_connectivity_rejects_unlisted_route_without_probe() -> None:
    called: list[str] = []

    def probe(route: str) -> bool:
        called.append(route)
        return True

    manager = ConnectivityManager({"approved"}, probe)
    assert not manager.may_proceed("unknown")
    assert called == []


def test_connectivity_fails_closed_on_probe_exception() -> None:
    def broken_probe(_route: str) -> bool:
        raise RuntimeError("mock probe failure")

    manager = ConnectivityManager({"approved"}, broken_probe)
    assert not manager.may_proceed("approved")


def test_reporting_is_disabled_by_default() -> None:
    engine = ReportingEngine()
    report = DailyReport("2026-10-21", "mock only", {"trades": 0.0})
    assert "mock only" in engine.render_text(report)
    assert not engine.deliver(report)


def test_strategy_scaffold_generates_no_entry_or_exit_signals() -> None:
    strategy = AliXStrategy()
    dataframe = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [12.0, 12.0],
            "low": [9.0, 10.0],
            "close": [11.0, 11.5],
            "volume": [5.0, 6.0],
        }
    )
    metadata: dict[str, Any] = {}
    enriched = strategy.populate_indicators(dataframe.copy(), metadata)
    entries = strategy.populate_entry_trend(enriched.copy(), metadata)
    exits = strategy.populate_exit_trend(enriched.copy(), metadata)
    assert entries["enter_long"].eq(0).all()
    assert exits["exit_long"].eq(0).all()


def test_data_quality_rejects_missing_columns_and_invalid_numbers() -> None:
    missing = validate_candles([{"date": 1000}])
    assert not missing.passed
    assert any("missing_columns" in reason for reason in missing.reasons)

    invalid = candle(1000)
    invalid["open"] = float("nan")
    invalid["volume"] = -1.0
    result = validate_candles([invalid])
    assert not result.passed
    assert any("non_finite_open" in reason for reason in result.reasons)
    assert any("negative_volume" in reason for reason in result.reasons)


def test_data_quality_rejects_stale_candles() -> None:
    result = validate_candles(
        [candle(1000)], max_age_seconds=60, now_epoch=2000
    )
    assert not result.passed
    assert "stale_latest_candle" in result.reasons
