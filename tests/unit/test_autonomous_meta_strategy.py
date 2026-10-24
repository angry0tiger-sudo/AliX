from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

import services.ai_core.autonomous_meta as autonomous
from services.ai_core.autonomous_meta import (
    STRATEGY_IDS,
    StrategyRegistry,
    StrategySelection,
    WalkForwardConfig,
    choose_strategy,
    compute_autonomous_features,
    evaluate_historical_bars,
    generate_entry_signal,
    load_strategy_registry,
    make_research_decision,
    make_walk_forward_folds,
    plan_dynamic_risk,
    summarize_policy_metrics,
)


def hourly_candles(rows: int = 420, *, positive_volume: bool = True) -> DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100.0 * np.exp(
        np.cumsum(0.00015 + 0.0025 * np.sin(index / 8.0) + 0.001 * np.sin(index / 2.7))
    )
    opened = np.r_[close[0] * 0.999, close[:-1]]
    high = np.maximum(opened, close) * (1.0015 + 0.0001 * (index % 3))
    low = np.minimum(opened, close) * (0.9985 - 0.0001 * (index % 2))
    volume = 100.0 + 25.0 * np.sin(index / 5.0) if positive_volume else np.zeros(rows)
    volume = np.maximum(volume, 1.0) if positive_volume else volume
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01T00:00:00Z", periods=rows, freq="1h"),
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def signal_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "close": 110.0,
        "ema_fast": 109.0,
        "ema_slow": 108.0,
        "macd": 2.0,
        "macd_signal": 1.0,
        "rsi_14": 20.0,
        "bollinger_lower": 111.0,
        "bollinger_upper": 115.0,
        "previous_high_20": 109.0,
        "previous_low_20": 101.0,
        "atr_14": 2.0,
        "volume": 200.0,
        "volume_mean_20": 100.0,
        "relative_volume": 2.0,
        "return_1h": 0.01,
        "momentum_3h": 0.02,
    }
    row.update(updates)
    return row


@pytest.fixture()
def strategy_registry() -> StrategyRegistry:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "research"
        / "autonomous-meta-strategy-registry.json"
    )
    return load_strategy_registry(path)


def test_registry_has_four_independent_research_only_strategies(
    strategy_registry: StrategyRegistry,
) -> None:
    assert (
        tuple(spec.strategy_id for spec in strategy_registry.strategies) == STRATEGY_IDS
    )
    assert all(
        spec.lifecycle == "RESEARCH_ONLY" for spec in strategy_registry.strategies
    )
    assert all(
        not spec.owner_approved and not spec.allow_short
        for spec in strategy_registry.strategies
    )
    assert (
        strategy_registry.code_sha256
        == hashlib.sha256(Path(autonomous.__file__).read_bytes()).hexdigest()
    )
    assert strategy_registry.document_sha256


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    (
        ("invalid_json", "autonomous_registry_json_invalid"),
        ("wrong_root", "autonomous_registry_schema_invalid"),
        ("bad_hash", "autonomous_registry_code_hash_mismatch"),
        ("missing_strategy", "autonomous_registry_strategy_count_invalid"),
        ("missing_entry_field", "autonomous_registry_entry_schema_invalid"),
        ("approval", "autonomous_strategy_must_remain_research_only"),
        ("duplicate_id", "autonomous_registry_duplicate_strategy_ids"),
        ("wrong_regime", "autonomous_strategy_regime_invalid"),
        ("bad_list", "autonomous_registry_lists_invalid"),
        ("bad_relative_volume", "autonomous_registry_relative_volume_invalid"),
    ),
)
def test_registry_loader_fails_closed(
    strategy_registry: StrategyRegistry,
    mutation: str,
    expected_reason: str,
) -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "research"
        / "autonomous-meta-strategy-registry.json"
    )
    raw = path.read_text(encoding="utf-8")
    if mutation == "invalid_json":
        document = "{"
    else:
        decoded = json.loads(raw)
        if mutation == "wrong_root":
            decoded["unexpected"] = True
        elif mutation == "bad_hash":
            decoded["code_sha256"] = "0" * 64
        elif mutation == "missing_strategy":
            decoded["strategies"].pop()
        elif mutation == "missing_entry_field":
            decoded["strategies"][0].pop("version")
        elif mutation == "approval":
            decoded["strategies"][0]["owner_approved"] = True
        elif mutation == "duplicate_id":
            decoded["strategies"][1]["strategy_id"] = decoded["strategies"][0][
                "strategy_id"
            ]
        elif mutation == "wrong_regime":
            decoded["strategies"][0]["eligible_regimes"] = ["not_a_regime"]
        elif mutation == "bad_list":
            decoded["strategies"][0]["required_features"] = "close"
        elif mutation == "bad_relative_volume":
            decoded["strategies"][0]["minimum_relative_volume"] = "high"
        document = json.dumps(decoded)
    if mutation == "bad_hash":
        with pytest.raises(ValueError, match=expected_reason):
            autonomous._parse_registry_document(
                document,
                hashlib.sha256(Path(autonomous.__file__).read_bytes()).hexdigest(),
            )
    elif mutation == "invalid_json":
        with pytest.raises(ValueError, match=expected_reason):
            autonomous._parse_registry_document(document, strategy_registry.code_sha256)
    else:
        with pytest.raises(ValueError, match=expected_reason):
            autonomous._parse_registry_document(document, strategy_registry.code_sha256)


def test_registry_file_missing_and_registry_regime_contract_rejected(
    tmp_path: Path,
    strategy_registry: StrategyRegistry,
) -> None:
    with pytest.raises(ValueError, match="registry_file_missing"):
        load_strategy_registry(tmp_path / "missing.json")
    changed = replace(
        strategy_registry.strategies[0],
        eligible_regimes=("range",),
    )
    invalid = StrategyRegistry(
        revision=strategy_registry.revision,
        code_sha256=strategy_registry.code_sha256,
        document_sha256=strategy_registry.document_sha256,
        strategies=tuple(
            changed if item.strategy_id == changed.strategy_id else item
            for item in strategy_registry.strategies
        ),
    )
    selection = choose_strategy(
        "trend", 0.9, signal_row(), invalid, source_volume_available=True
    )
    assert selection.status == "NO_TRADE"
    assert selection.strategy_id is None
    assert selection.reason == "regime_has_no_valid_candidate"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("lifecycle", "SELECTABLE", "must_remain_research_only"),
        ("owner_approved", True, "must_remain_research_only"),
        ("allow_short", True, "must_remain_research_only"),
        ("timeframe", "4h", "contract_invalid"),
        ("eligible_regimes", (), "contract_invalid"),
        ("required_features", (), "features_or_horizon_invalid"),
        ("max_holding_bars", 0, "features_or_horizon_invalid"),
        ("minimum_relative_volume", float("nan"), "parameters_non_finite"),
        ("atr_stop_multiple", 0.0, "parameters_out_of_range"),
    ),
)
def test_strategy_spec_rejects_invalid_contracts(
    strategy_registry: StrategyRegistry,
    field: str,
    value: object,
    reason: str,
) -> None:
    template = strategy_registry.strategies[0]
    with pytest.raises(ValueError, match=reason):
        if field == "lifecycle":
            replace(template, lifecycle=cast(str, value))
        elif field == "owner_approved":
            replace(template, owner_approved=cast(bool, value))
        elif field == "allow_short":
            replace(template, allow_short=cast(bool, value))
        elif field == "timeframe":
            replace(template, timeframe=cast(str, value))
        elif field == "eligible_regimes":
            replace(template, eligible_regimes=cast(tuple[str, ...], value))
        elif field == "required_features":
            replace(template, required_features=cast(tuple[str, ...], value))
        elif field == "max_holding_bars":
            replace(template, max_holding_bars=cast(int, value))
        elif field == "minimum_relative_volume":
            replace(template, minimum_relative_volume=cast(float, value))
        elif field == "atr_stop_multiple":
            replace(template, atr_stop_multiple=cast(float, value))
        else:
            raise AssertionError(f"unexpected strategy field: {field}")


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("initial_train_rows", 99, "config_invalid"),
        ("fold_count", 0, "config_invalid"),
        ("minimum_confidence", 0.49, "out_of_range"),
        ("fee_bps_per_side", -1.0, "out_of_range"),
        ("slippage_bps_per_side", float("nan"), "non_finite"),
        ("risk_fraction", 0.03, "out_of_range"),
        ("max_position_fraction", 0.0, "out_of_range"),
        ("initial_equity", 0.0, "out_of_range"),
    ),
)
def test_walk_forward_config_rejects_invalid_risk_or_split_values(
    field: str, value: object, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        if field == "initial_train_rows":
            WalkForwardConfig(initial_train_rows=cast(int, value))
        elif field == "fold_count":
            WalkForwardConfig(fold_count=cast(int, value))
        elif field == "minimum_confidence":
            WalkForwardConfig(minimum_confidence=cast(float, value))
        elif field == "fee_bps_per_side":
            WalkForwardConfig(fee_bps_per_side=cast(float, value))
        elif field == "slippage_bps_per_side":
            WalkForwardConfig(slippage_bps_per_side=cast(float, value))
        elif field == "risk_fraction":
            WalkForwardConfig(risk_fraction=cast(float, value))
        elif field == "max_position_fraction":
            WalkForwardConfig(max_position_fraction=cast(float, value))
        elif field == "initial_equity":
            WalkForwardConfig(initial_equity=cast(float, value))
        else:
            raise AssertionError(f"unexpected walk-forward field: {field}")


def test_features_are_causal_and_atr_scales_with_price_ranges() -> None:
    candles = hourly_candles()
    features = compute_autonomous_features(candles)
    altered = candles.copy()
    for column in ("close", "high", "open", "low"):
        values = pd.to_numeric(altered[column], errors="raise").to_numpy(dtype=float)
        values[-1] = values[-1] * 1.4
        altered[column] = values
    changed_features = compute_autonomous_features(altered)
    pd.testing.assert_series_equal(
        features.iloc[:-1]["atr_14"],
        changed_features.iloc[:-1]["atr_14"],
        check_names=True,
    )
    assert features["macd"].notna().sum() > 0
    assert features["rsi_14"].notna().sum() > 0
    assert features["previous_high_20"].notna().sum() > 0
    assert features["relative_volume"].notna().sum() > 0


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("missing_column", "columns_missing"),
        ("empty", "empty_or_duplicate"),
        ("duplicate_date", "duplicate_or_non_monotonic"),
        ("descending", "duplicate_or_non_monotonic"),
        ("gap", "not_contiguous"),
        ("naive_time", "timezone_aware"),
        ("non_finite", "non_finite"),
        ("negative_volume", "out_of_range"),
        ("bad_ohlc", "ohlc_invariant"),
    ),
)
def test_hourly_market_data_validation_fails_closed(mutation: str, reason: str) -> None:
    candles = hourly_candles(80)
    if mutation == "missing_column":
        candles = candles.drop(columns=["low"])
    elif mutation == "empty":
        candles = candles.iloc[:0]
    elif mutation == "duplicate_date":
        candles.loc[1, "date"] = candles.loc[0, "date"]
    elif mutation == "descending":
        candles = candles.iloc[::-1].reset_index(drop=True)
    elif mutation == "gap":
        date_values = pd.to_datetime(candles["date"], utc=True).tolist()
        date_values[20] = pd.Timestamp(date_values[20]) + pd.Timedelta(1, unit="m")
        candles["date"] = pd.to_datetime(date_values, utc=True)
    elif mutation == "naive_time":
        candles["date"] = candles["date"].dt.tz_localize(None)
    elif mutation == "non_finite":
        candles.loc[10, "close"] = np.inf
    elif mutation == "negative_volume":
        candles.loc[10, "volume"] = -1.0
    elif mutation == "bad_ohlc":
        low_values = pd.to_numeric(candles["low"], errors="raise").to_numpy(dtype=float)
        high_values = pd.to_numeric(candles["high"], errors="raise").to_numpy(
            dtype=float
        )
        high_values[10] = low_values[10] - 1.0
        candles["high"] = high_values
    with pytest.raises(ValueError, match=reason):
        compute_autonomous_features(candles)


@pytest.mark.parametrize(
    ("strategy_id", "row"),
    (
        (
            "trend_following",
            {
                "close": 110.0,
                "ema_fast": 109.0,
                "ema_slow": 108.0,
                "macd": 2.0,
                "macd_signal": 1.0,
            },
        ),
        ("mean_reversion", {"close": 80.0, "rsi_14": 20.0, "bollinger_lower": 81.0}),
        ("volatility_breakout", {"close": 105.0, "previous_high_20": 104.0}),
        (
            "scalping",
            {
                "close": 105.0,
                "volume": 300.0,
                "relative_volume": 2.0,
                "ema_fast": 104.0,
                "momentum_3h": 0.01,
            },
        ),
    ),
)
def test_four_independent_strategies_have_distinct_entry_conditions(
    strategy_id: str, row: dict[str, object]
) -> None:
    assert generate_entry_signal(
        strategy_id, row, source_volume_available=True, minimum_relative_volume=1.5
    )


def test_strategy_signals_reject_unknown_missing_and_volume_unverified_inputs() -> None:
    with pytest.raises(ValueError, match="strategy_id_unknown"):
        generate_entry_signal("unregistered", {}, source_volume_available=True)
    assert not generate_entry_signal(
        "trend_following", {}, source_volume_available=True
    )
    assert not generate_entry_signal(
        "scalping", signal_row(), source_volume_available=False
    )
    assert not generate_entry_signal(
        "scalping", signal_row(relative_volume=1.2), source_volume_available=True
    )
    assert not generate_entry_signal(
        "mean_reversion", signal_row(rsi_14=55.0), source_volume_available=True
    )
    assert not generate_entry_signal(
        "volatility_breakout",
        signal_row(close=105.0, previous_high_20=106.0),
        source_volume_available=True,
    )


@pytest.mark.parametrize(
    ("regime", "row", "volume_available", "expected"),
    (
        ("trend", signal_row(), True, "trend_following"),
        ("high_volatility", signal_row(), True, "volatility_breakout"),
        ("range", signal_row(), False, "mean_reversion"),
        ("range", signal_row(relative_volume=2.0, volume=20.0), True, "scalping"),
        ("range", signal_row(relative_volume=1.1), True, "mean_reversion"),
    ),
)
def test_hmm_selector_routes_by_regime_confidence_and_data_eligibility(
    strategy_registry: StrategyRegistry,
    regime: str,
    row: dict[str, object],
    volume_available: bool,
    expected: str,
) -> None:
    selection = choose_strategy(
        regime, 0.9, row, strategy_registry, source_volume_available=volume_available
    )
    assert selection.status == "SELECTED_RESEARCH_ONLY"
    assert selection.strategy_id == expected
    assert selection.order_permission is False
    assert selection.champion is None


@pytest.mark.parametrize(
    ("regime", "confidence", "expected_reason"),
    (
        (None, 0.9, "regime_missing_or_unknown"),
        ("unknown", 0.9, "regime_missing_or_unknown"),
        ("trend", None, "confidence_invalid"),
        ("trend", float("nan"), "confidence_invalid"),
        ("trend", 0.54, "confidence_below_threshold"),
    ),
)
def test_hmm_selector_abstains_on_bad_or_uncertain_regimes(
    strategy_registry: StrategyRegistry,
    regime: str | None,
    confidence: float | None,
    expected_reason: str,
) -> None:
    selection = choose_strategy(
        regime,
        confidence,
        signal_row(),
        strategy_registry,
        source_volume_available=True,
    )
    assert selection.status == "NO_TRADE"
    assert selection.strategy_id is None
    assert expected_reason in selection.reason


def test_selector_rejects_registry_regime_mismatch(
    strategy_registry: StrategyRegistry,
) -> None:
    trend = next(
        item
        for item in strategy_registry.strategies
        if item.strategy_id == "trend_following"
    )
    invalid_trend = replace(trend, eligible_regimes=("range",))
    registry = StrategyRegistry(
        revision=strategy_registry.revision,
        code_sha256=strategy_registry.code_sha256,
        document_sha256=strategy_registry.document_sha256,
        strategies=tuple(
            invalid_trend if item.strategy_id == trend.strategy_id else item
            for item in strategy_registry.strategies
        ),
    )
    decision = choose_strategy(
        "trend", 0.9, signal_row(), registry, source_volume_available=True
    )
    assert decision.status == "NO_TRADE"
    assert decision.reason == "regime_has_no_valid_candidate"


def test_research_decision_separates_selection_from_signal(
    strategy_registry: StrategyRegistry,
) -> None:
    selected_but_flat = make_research_decision(
        "trend",
        0.9,
        signal_row(ema_fast=107.0, ema_slow=108.0),
        strategy_registry,
        source_volume_available=True,
    )
    assert selected_but_flat.status == "NO_SIGNAL"
    assert selected_but_flat.strategy_id == "trend_following"
    signal = make_research_decision(
        "trend", 0.9, signal_row(), strategy_registry, source_volume_available=True
    )
    assert signal.status == "SIGNAL_RESEARCH_ONLY"
    with pytest.raises(ValueError, match="cannot_authorize_or_promote"):
        StrategySelection("trend", 0.9, "trend_following", "SELECTED", "test", True)


def test_dynamic_risk_plan_scales_with_atr_confidence_and_regime(
    strategy_registry: StrategyRegistry,
) -> None:
    spec = next(
        item
        for item in strategy_registry.strategies
        if item.strategy_id == "trend_following"
    )
    config = WalkForwardConfig(initial_train_rows=120, fold_count=2)
    calm = plan_dynamic_risk(
        entry_price=100.0,
        atr=2.0,
        regime="range",
        confidence=0.8,
        equity=10_000.0,
        spec=spec,
        config=config,
    )
    volatile = plan_dynamic_risk(
        entry_price=100.0,
        atr=2.0,
        regime="high_volatility",
        confidence=0.9,
        equity=10_000.0,
        spec=spec,
        config=config,
    )
    assert calm is not None and volatile is not None
    assert volatile.stop_distance > calm.stop_distance
    assert volatile.target_distance > calm.target_distance
    assert volatile.risk_budget > calm.risk_budget
    assert calm.stop_loss < calm.entry_price < calm.take_profit
    assert calm.notional <= 10_000.0 * config.max_position_fraction
    with pytest.raises(ValueError, match="cannot_authorize_orders"):
        replace(calm, order_permission=True)


@pytest.mark.parametrize(
    ("kwargs", "config"),
    (
        (
            {
                "entry_price": 0.0,
                "atr": 1.0,
                "regime": "trend",
                "confidence": 0.9,
                "equity": 1000.0,
            },
            None,
        ),
        (
            {
                "entry_price": 100.0,
                "atr": 0.0,
                "regime": "trend",
                "confidence": 0.9,
                "equity": 1000.0,
            },
            None,
        ),
        (
            {
                "entry_price": 100.0,
                "atr": 1.0,
                "regime": "trend",
                "confidence": 0.4,
                "equity": 1000.0,
            },
            None,
        ),
        (
            {
                "entry_price": float("nan"),
                "atr": 1.0,
                "regime": "trend",
                "confidence": 0.9,
                "equity": 1000.0,
            },
            None,
        ),
    ),
)
def test_dynamic_risk_plan_abstains_on_invalid_sizing_inputs(
    strategy_registry: StrategyRegistry,
    kwargs: dict[str, Any],
    config: WalkForwardConfig | None,
) -> None:
    spec = strategy_registry.strategies[0]
    selected_config = config or WalkForwardConfig(initial_train_rows=120, fold_count=2)
    assert plan_dynamic_risk(**kwargs, spec=spec, config=selected_config) is None


def test_walk_forward_splits_are_expanding_and_insufficient_history_fails() -> None:
    config = WalkForwardConfig(initial_train_rows=120, fold_count=2)
    assert make_walk_forward_folds(420, config) == ((120, 120, 270), (270, 270, 420))
    with pytest.raises(ValueError, match="rows_insufficient"):
        make_walk_forward_folds(170, config)


def test_exit_simulation_uses_stop_first_when_both_barriers_hit(
    strategy_registry: StrategyRegistry,
) -> None:
    bars = hourly_candles(80)
    spec = strategy_registry.strategies[0]
    config = WalkForwardConfig(initial_train_rows=120, fold_count=2)
    plan = plan_dynamic_risk(
        entry_price=100.0,
        atr=1.0,
        regime="trend",
        confidence=0.9,
        equity=10_000.0,
        spec=spec,
        config=config,
    )
    assert plan is not None
    bars.loc[30, ["open", "high", "low", "close"]] = [100.0, 104.0, 97.0, 101.0]
    exit_index, exit_price, reason = autonomous._exit_fill(
        bars, entry_index=30, test_stop=70, plan=plan, max_holding_bars=3
    )
    assert exit_index == 30
    assert exit_price == plan.stop_loss
    assert reason == "STOP_LOSS_CONSERVATIVE_TIE_BREAK"


@pytest.mark.parametrize(
    ("bar_values", "expected_reason"),
    (
        ([95.0, 101.0, 94.0, 96.0], "STOP_GAP"),
        ([105.0, 107.0, 104.0, 106.0], "TAKE_PROFIT_GAP"),
    ),
)
def test_exit_simulation_handles_gap_fills(
    strategy_registry: StrategyRegistry,
    bar_values: list[float],
    expected_reason: str,
) -> None:
    bars = hourly_candles(80)
    spec = strategy_registry.strategies[0]
    config = WalkForwardConfig(initial_train_rows=120, fold_count=2)
    plan = plan_dynamic_risk(
        entry_price=100.0,
        atr=1.0,
        regime="trend",
        confidence=0.9,
        equity=10_000.0,
        spec=spec,
        config=config,
    )
    assert plan is not None
    bars.loc[30, ["open", "high", "low", "close"]] = bar_values
    _, _, reason = autonomous._exit_fill(
        bars, entry_index=30, test_stop=70, plan=plan, max_holding_bars=3
    )
    assert reason == expected_reason


def test_metrics_and_comparisons_report_missing_samples_without_inventing_values() -> (
    None
):
    no_trades = summarize_policy_metrics([], [10_000.0], initial_equity=10_000.0)
    assert no_trades["status"] == "NO_TRADES"
    assert no_trades["win_rate_pct"] is None
    assert no_trades["expectancy_bps_per_trade"] is None
    trades = [
        {
            "strategy_id": "trend_following",
            "net_return_bps": 100.0,
            "net_pnl_quote": 10.0,
        },
        {
            "strategy_id": "mean_reversion",
            "net_return_bps": -40.0,
            "net_pnl_quote": -4.0,
        },
    ]
    metrics = summarize_policy_metrics(
        trades, [10_000.0, 10_010.0, 10_006.0], initial_equity=10_000.0
    )
    assert metrics["win_rate_pct"] == 50.0
    assert metrics["expectancy_bps_per_trade"] == 30.0
    assert metrics["max_drawdown_pct"] > 0
    comparisons = autonomous._comparisons(metrics, {"scalping": no_trades})
    assert comparisons["scalping"]["status"] == "INSUFFICIENT_TRADES_FOR_COMPARISON"
    assert comparisons["scalping"]["improved_both"] is None


def test_historical_walk_forward_emits_four_baselines_and_closed_safety_contracts(
    strategy_registry: StrategyRegistry,
) -> None:
    bars = hourly_candles(420, positive_volume=True)
    config = WalkForwardConfig(
        initial_train_rows=180,
        fold_count=2,
        minimum_confidence=0.5,
        initial_equity=10_000.0,
    )
    report, trades, decisions = evaluate_historical_bars(
        bars,
        strategy_registry,
        commit_sha="a" * 40,
        source={"volume_observed": True, "final_holdout_used": False},
        config=config,
    )
    assert report["run_status"] == "COMPLETED_RESEARCH_ONLY"
    assert report["walk_forward"]["fold_count"] == 2
    assert report["walk_forward"]["final_holdout_used"] is False
    assert report["strategy_registry"]["candidate_count"] == 4
    assert set(report["economic_evaluation"]["fixed_strategy_baselines"]) == set(
        STRATEGY_IDS
    )
    assert report["economic_evaluation"]["economic_acceptance"] == "NO-GO"
    assert all(item["order_permission"] is False for item in trades)
    assert all(item["real_order_attempted"] is False for item in trades)
    assert decisions
    assert all(item["order_permission"] is False for item in decisions)
    safety = report["safety_invariants"]
    assert safety["Champion"] is None
    assert safety["order_permission"] is False
    assert safety["trading_authorization"] == "NO-GO"
    assert safety["private_api_calls"] == 0
    assert safety["real_order_attempts"] == 0


def test_historical_evaluation_blocks_scalping_without_positive_source_volume(
    strategy_registry: StrategyRegistry,
) -> None:
    bars = hourly_candles(420, positive_volume=False)
    report, trades, _ = evaluate_historical_bars(
        bars,
        strategy_registry,
        commit_sha="b" * 40,
        source={"volume_observed": True, "final_holdout_used": False},
        config=WalkForwardConfig(
            initial_train_rows=180, fold_count=2, minimum_confidence=0.5
        ),
    )
    assert (
        report["strategy_registry"]["scalping_status"]
        == "BLOCKED_NO_POSITIVE_SOURCE_VOLUME"
    )
    assert all(trade["strategy_id"] != "scalping" for trade in trades)
    assert report["safety_invariants"]["Champion"] is None


def test_evaluation_rejects_invalid_commit_sha(
    strategy_registry: StrategyRegistry,
) -> None:
    with pytest.raises(ValueError, match="commit_sha_invalid"):
        evaluate_historical_bars(
            hourly_candles(100),
            strategy_registry,
            commit_sha="invalid",
            source={},
            config=WalkForwardConfig(initial_train_rows=120, fold_count=2),
        )


def test_trade_cost_assumptions_are_deducted_from_net_results() -> None:
    config = WalkForwardConfig(initial_train_rows=120, fold_count=2)
    pnl, costs = autonomous._net_trade_result(100.0, 102.0, 1.0, config)
    assert costs > 0
    assert pnl < 2.0
    assert pnl == pytest.approx(2.0 - costs)
