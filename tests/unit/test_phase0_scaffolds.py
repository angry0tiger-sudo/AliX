from connectivity.manager.manager import ConnectivityManager
from services.data_quality.gate import validate_candles
from services.reporting.engine import DailyReport, ReportingEngine


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
