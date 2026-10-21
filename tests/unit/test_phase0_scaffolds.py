from services.data_quality.gate import validate_candles
from connectivity.manager.manager import ConnectivityManager
from services.reporting.engine import DailyReport, ReportingEngine

def candle(ts=1000):
    return {"date": ts, "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "volume": 5.0}

def test_data_quality_accepts_valid_candles():
    assert validate_candles([candle(1000), candle(1060)]).passed

def test_data_quality_fails_closed_for_empty_data():
    result = validate_candles([])
    assert not result.passed and "empty_dataset" in result.reasons

def test_data_quality_rejects_bad_ohlc_and_order():
    bad = candle(1000)
    bad["high"] = 8
    result = validate_candles([candle(1000), bad])
    assert not result.passed
    assert any("invalid_ohlc_range" in reason for reason in result.reasons)
    assert any("timestamp_not_strictly_increasing" in reason for reason in result.reasons)

def test_connectivity_rejects_unlisted_route_without_probe():
    called = []
    manager = ConnectivityManager({"approved"}, lambda route: called.append(route) or True)
    assert not manager.may_proceed("unknown")
    assert called == []

def test_connectivity_fails_closed_on_probe_exception():
    manager = ConnectivityManager({"approved"}, lambda route: (_ for _ in ()).throw(RuntimeError()))
    assert not manager.may_proceed("approved")

def test_reporting_is_disabled_by_default():
    engine = ReportingEngine()
    report = DailyReport("2026-10-21", "mock only", {"trades": 0.0})
    assert "mock only" in engine.render_text(report)
    assert not engine.deliver(report)
