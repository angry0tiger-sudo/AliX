"""Mock-only tests for M7 cross-sectional portfolio mechanics."""

from __future__ import annotations

import math

import pytest

from scripts.run_m7_cross_sectional import portfolio_interval_return, select_top_two


def test_select_top_two_uses_only_positive_predictions_and_stable_ties() -> None:
    predictions = {"ADAUSDT": 0.04, "ETHUSDT": 0.04, "SOLUSDT": -0.01, "XRPUSDT": 0.02}
    assert select_top_two(predictions) == ["ADAUSDT", "ETHUSDT"]


def test_select_top_two_returns_at_most_two_assets() -> None:
    assert select_top_two({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}) == ["D", "C"]


def test_select_top_two_returns_empty_when_no_score_is_positive() -> None:
    assert select_top_two({"BTC": 0.0, "ETH": -0.1}) == []


def test_select_top_two_ignores_non_finite_scores() -> None:
    assert select_top_two({"A": float("nan"), "B": float("inf"), "C": 0.01}) == ["C"]


def test_portfolio_interval_return_equal_weights_and_round_trip_cost() -> None:
    total, trade_pnls = portfolio_interval_return(["ETHUSDT", "ADAUSDT"], {"ETHUSDT": 0.04, "ADAUSDT": 0.02}, 50)
    assert trade_pnls == pytest.approx([0.5 * (0.04 - 0.005), 0.5 * (0.02 - 0.005)])
    assert total == pytest.approx(sum(trade_pnls))


def test_portfolio_interval_return_keeps_unused_half_in_usdt() -> None:
    total, trade_pnls = portfolio_interval_return(["ETHUSDT"], {"ETHUSDT": 0.10}, 0)
    assert trade_pnls == pytest.approx([0.05])
    assert total == pytest.approx(0.05)


def test_portfolio_interval_return_empty_selection_is_cash() -> None:
    assert portfolio_interval_return([], {}, 90) == (0.0, [])


def test_portfolio_interval_return_rejects_more_than_two_assets() -> None:
    with pytest.raises(ValueError, match="m7_invalid_selected_symbol_set"):
        portfolio_interval_return(["A", "B", "C"], {"A": 0.1, "B": 0.1, "C": 0.1}, 50)


def test_portfolio_interval_return_rejects_missing_realized_return() -> None:
    with pytest.raises(ValueError, match="m7_missing_realized_return"):
        portfolio_interval_return(["ETHUSDT"], {}, 50)


def test_portfolio_interval_return_rejects_non_finite_return() -> None:
    with pytest.raises(ValueError, match="m7_invalid_realized_return"):
        portfolio_interval_return(["ETHUSDT"], {"ETHUSDT": math.nan}, 50)
