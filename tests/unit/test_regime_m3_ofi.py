from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from scripts.run_regime_m3_ofi import (
    Book,
    Candidate,
    Trade,
    _book_features,
    _event_ofi,
    _make_pipeline,
    _max_drawdown,
    _net_profit_factor,
    _verdict,
)

UTC = timezone.utc


def _book(time: datetime, bid: float = 100.0, bid_qty: float = 10.0, ask: float = 102.0, ask_qty: float = 20.0) -> Book:
    bids = tuple((bid - index, bid_qty + index) for index in range(5))
    asks = tuple((ask + index, ask_qty + index) for index in range(5))
    return Book(time, "websocket", 0, bids, asks)


def _trade(pnl: float, minute: int) -> Trade:
    entry = datetime(2026, 11, 27, 0, minute, tzinfo=UTC)
    return Trade(entry, entry.replace(hour=1), 1.0 / 60.0, 100.0, pnl + 0.009, pnl, pnl / 60.0)


def test_book_microprice_and_top5_imbalance_match_registered_formulas() -> None:
    book = _book(datetime(2026, 11, 27, 0, 0, tzinfo=UTC))

    microprice_bps, imbalance = _book_features(book)

    midpoint = 101.0
    expected_microprice = (102.0 * 10.0 + 100.0 * 20.0) / 30.0
    assert microprice_bps == pytest.approx(10_000.0 * (expected_microprice - midpoint) / midpoint)
    bid_depth = sum(10.0 + index for index in range(5))
    ask_depth = sum(20.0 + index for index in range(5))
    assert imbalance == pytest.approx((bid_depth - ask_depth) / (bid_depth + ask_depth))


def test_best_level_and_multilevel_ofi_are_five_registered_rank_values() -> None:
    previous = _book(datetime(2026, 11, 27, 0, 0, tzinfo=UTC))
    current = _book(datetime(2026, 11, 27, 0, 1, tzinfo=UTC), bid=101.0, bid_qty=12.0, ask=103.0, ask_qty=18.0)

    values = _event_ofi(previous, current)

    assert len(values) == 5
    assert all(value > 0.0 for value in values)


def test_standard_scaler_is_fitted_on_train_only() -> None:
    config = {
        "model": {
            "parameters": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "fit_intercept": True,
                "class_weight": None,
                "max_iter": 1000,
                "tol": 0.0001,
                "random_state": 42,
            }
        }
    }
    train = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    labels = np.asarray([0, 0, 1, 1])
    test = np.asarray([[1_000_000.0, -1_000_000.0]])
    model = _make_pipeline(config)
    model.fit(train, labels)
    scaler = model.named_steps["standard_scaler"]

    assert scaler.mean_.tolist() == pytest.approx(train.mean(axis=0).tolist())
    assert scaler.mean_.tolist() != pytest.approx(np.vstack([train, test]).mean(axis=0).tolist())


def test_net_profit_factor_uses_net_notional_pnl() -> None:
    trades = [_trade(0.011, 0), _trade(-0.029, 1)]

    assert _net_profit_factor(trades) == pytest.approx(0.3793103448)


def test_net_profit_factor_is_undefined_without_losses() -> None:
    assert _net_profit_factor([_trade(0.01, 0)]) is None
    assert _net_profit_factor([]) is None


def test_max_drawdown_uses_peak_to_trough_nav() -> None:
    assert _max_drawdown([1.0, 1.1, 1.045, 1.12]) == pytest.approx(0.05)


def test_gate_requires_every_registered_economic_condition() -> None:
    metrics = {"completed_oos_trades": 120, "net_profit_factor": 1.2, "max_drawdown": 0.10}

    verdict, gates = _verdict(metrics, stable_folds=3, eligible_folds=4)

    assert verdict == "PASS"
    assert all(gates.values())


def test_insufficient_trade_count_is_not_promoted_by_high_profit_factor() -> None:
    metrics = {"completed_oos_trades": 99, "net_profit_factor": 2.0, "max_drawdown": 0.01}

    verdict, gates = _verdict(metrics, stable_folds=4, eligible_folds=4)

    assert verdict == "INSUFFICIENT_SAMPLE / NO-GO"
    assert gates["completed_oos_trades_gte_100"] is False


def test_candidate_contract_has_exactly_eight_features() -> None:
    row = Candidate(
        datetime(2026, 11, 27, 0, 0, tzinfo=UTC),
        (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -0.2, 0.3),
        100.0,
        101.0,
        1,
    )

    assert len(row.features) == 8
    assert row.label == 1
