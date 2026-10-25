from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_deap_evolution import (
    DIRECTIONAL_EXIT,
    EMA_ENTRY,
    EXIT_MODES,
    FASTS,
    MAX_DRAWDOWN,
    MIN_REQUIRED_TIMEFRAME_POPULATION,
    REQUIRED_TIMEFRAMES,
    SLOWS,
    TRAILING_ACTIVATION,
    CandidateEvaluation,
    EvaluationMetrics,
    TIMEFRAMES,
    _build_features,
    _economic_assessment,
    _ensure_timeframe_coverage,
    _resample,
    _strategy_genome,
    _thresholds,
    build_position_states,
    fitness_score,
    metrics_from_positions,
    repair_individual,
)
from services.ai_core.self_evolving_genome import genome_sha256


def _metrics(
    trades: int,
    *,
    expectancy: float | None = 0.002,
    net_return: float = 0.05,
    drawdown: float = 0.10,
    win_rate: float = 0.55,
    profit_factor: float | None = 1.8,
    gross_profit: float = 0.08,
    gross_loss: float = 0.03,
    missed_count: int = 0,
    missed_return: float = 0.0,
    volatility_return: float = 0.01,
    trend_return: float = 0.02,
) -> EvaluationMetrics:
    return EvaluationMetrics(
        trade_count=trades,
        win_rate=win_rate,
        net_expectancy=expectancy if trades else None,
        net_return=net_return,
        max_drawdown=drawdown,
        profit_factor=profit_factor,
        gross_profit=gross_profit if trades else 0.0,
        gross_loss=gross_loss if trades else 0.0,
        missed_upside_move_count=missed_count,
        missed_upside_return=missed_return,
        high_volatility_return=volatility_return,
        strong_trend_return=trend_return,
    )


def _candidate(
    training: EvaluationMetrics, validation: EvaluationMetrics
) -> CandidateEvaluation:
    reasons = []
    if training.trade_count == 0:
        reasons.append("training_has_no_completed_trades")
    if training.max_drawdown > MAX_DRAWDOWN:
        reasons.append("training_drawdown_above_limit")
    if validation.trade_count == 0:
        reasons.append("validation_has_no_completed_trades")
    if validation.profit_factor is None or validation.profit_factor <= 1.5:
        reasons.append("validation_profit_factor_not_above_1_5")
    if validation.net_return <= 0:
        reasons.append("validation_total_return_not_positive")
    return CandidateEvaluation(
        training=training,
        validation=validation,
        fitness=fitness_score(training),
        eligibility_reasons=tuple(reasons),
    )


def test_repair_keeps_all_evolved_genes_in_range_and_ema_periods_ordered() -> None:
    genes = [99, 1, 7, 5, 0, 4, 5, 3, 5, 99, 99]
    repaired = repair_individual(genes)
    assert len(repaired) == 11
    assert tuple(repaired[:2]) == (len(TIMEFRAMES) - 1, 1)
    assert repaired[2] == 0
    assert FASTS[repaired[3]] < SLOWS[repaired[4]]
    assert repaired[9] == len(TRAILING_ACTIVATION) - 1
    assert repaired[10] == len(EXIT_MODES) - 1


def test_repair_canonicalizes_inactive_ema_genes_for_other_indicators() -> None:
    genes = [0, 0, 7, 5, 7, 1, 2, 0, 1, 2, 1]
    repaired = repair_individual(genes)
    assert repaired[2] == 7
    assert repaired[3:5] == [0, 1]
    assert repaired[9:] == [2, 1]


def test_directional_thresholds_do_not_overlap_for_entry_and_exit() -> None:
    for entry_index in range(len(EMA_ENTRY)):
        for exit_index in range(len(DIRECTIONAL_EXIT)):
            genes = [0, 1, 0, 0, 1, entry_index, exit_index, 0, 0, 0, 0]
            assert _thresholds(genes)[1] < _thresholds(genes)[0]


def test_dynamic_trailing_stop_has_an_evolved_activation_threshold() -> None:
    close = np.array([100.0, 101.0, 103.0, 104.0, 101.0, 100.5, 100.0])
    entry = np.array([False, True, True, True, True, True, True])
    no_activation = build_position_states(
        entry, np.zeros(7, dtype=bool), close, 0.02, 0.0
    )
    delayed_activation = build_position_states(
        entry, np.zeros(7, dtype=bool), close, 0.02, 0.05
    )
    assert no_activation[1:].sum() < delayed_activation[1:].sum()


def test_invalid_dynamic_stop_parameters_fail_closed() -> None:
    with pytest.raises(ValueError, match="trailing_stop_parameters_invalid"):
        build_position_states(
            np.ones(3, dtype=bool), np.zeros(3, dtype=bool), np.ones(3), 1.0
        )


def test_profit_factor_is_gross_wins_over_gross_losses_after_costs() -> None:
    opens = np.array([100.0, 100.0, 90.0, 90.0, 110.0, 110.0, 100.0])
    closes = np.array([100.0, 110.0, 90.0, 100.0, 110.0, 100.0, 100.0])
    positions = np.array([True, False, True, False, True, False, False])
    metrics = metrics_from_positions(opens, closes, positions, 0.75, 15)
    assert metrics.trade_count >= 1
    assert metrics.gross_profit >= 0
    assert metrics.gross_loss >= 0
    if metrics.gross_loss > 0:
        assert metrics.profit_factor == pytest.approx(
            metrics.gross_profit / metrics.gross_loss
        )


def test_no_trade_candidate_reports_missed_upside_and_negative_fitness() -> None:
    opens = np.full(10, 100.0)
    closes = np.array([100, 100, 100, 106, 107, 105, 100, 101, 100, 100], dtype=float)
    positions = np.zeros(10, dtype=bool)
    metrics = metrics_from_positions(opens, closes, positions, 0.25, 1440)
    assert metrics.trade_count == 0
    assert metrics.net_expectancy is None
    assert metrics.profit_factor is None
    assert metrics.missed_upside_move_count >= 1
    assert fitness_score(metrics) < 0.0


def test_no_hard_trade_count_threshold_and_low_samples_are_scored_softly() -> None:
    one_trade = _metrics(1, profit_factor=3.0, gross_profit=0.03, gross_loss=0.01)
    ten_trades = _metrics(10, profit_factor=1.8, gross_profit=0.09, gross_loss=0.05)
    assert fitness_score(one_trade) > 0
    assert fitness_score(ten_trades) > 0
    assert fitness_score(_metrics(1)) != 0.0


def test_profit_factor_and_total_return_both_affect_fitness() -> None:
    low_pf = _metrics(
        20, profit_factor=1.1, gross_profit=0.11, gross_loss=0.10, net_return=0.05
    )
    high_pf = _metrics(
        20, profit_factor=2.0, gross_profit=0.20, gross_loss=0.10, net_return=0.05
    )
    high_return = _metrics(
        20, profit_factor=1.1, gross_profit=0.11, gross_loss=0.10, net_return=0.25
    )
    assert fitness_score(high_pf) > fitness_score(low_pf)
    assert fitness_score(high_return) > fitness_score(low_pf)


def test_opportunity_cost_penalty_reduces_fitness_for_missed_rallies() -> None:
    active = _metrics(12, missed_count=0, missed_return=0.0)
    idle = _metrics(
        0,
        expectancy=None,
        net_return=0.0,
        profit_factor=None,
        gross_profit=0,
        gross_loss=0,
        missed_count=3,
        missed_return=0.20,
        volatility_return=0,
        trend_return=0,
    )
    assert fitness_score(active) > fitness_score(idle)
    assert idle.missed_upside_move_count == 3


def test_regime_capture_bonus_rewards_trend_and_high_volatility_returns() -> None:
    base = _metrics(15, volatility_return=0.0, trend_return=0.0)
    regime_capturer = _metrics(15, volatility_return=0.10, trend_return=0.15)
    assert fitness_score(regime_capturer) > fitness_score(base)


def test_drawdown_over_15_percent_gets_severe_penalty() -> None:
    low_risk = _metrics(20, net_return=0.10, drawdown=0.10)
    high_risk = _metrics(20, net_return=0.80, drawdown=0.20)
    assert fitness_score(low_risk) > fitness_score(high_risk)


def test_economic_assessment_reports_profit_factor_target_without_hard_trade_minimums() -> (
    None
):
    training = _metrics(4, profit_factor=2.0, net_return=0.10, drawdown=0.05)
    validation = _metrics(2, profit_factor=1.7, net_return=0.04, drawdown=0.04)
    result = _economic_assessment(training, validation)
    assert result["minimum_completed_trades_enforced"] is False
    assert result["validation_profit_factor_above_1_5"] is True
    assert result["final_holdout_used"] is False
    assert result["production_promotion_eligible"] is False


def test_economic_assessment_fails_when_profit_factor_not_above_target() -> None:
    result = _economic_assessment(
        _metrics(4), _metrics(3, profit_factor=1.4, net_return=0.01)
    )
    assert result["status"] == "NO-GO"
    assert "validation_profit_factor_not_above_1_5" in result["reasons"]


def test_no_candidate_assessment_fails_closed() -> None:
    result = _economic_assessment(None, None)
    assert result["status"] == "NO-GO"
    assert result["minimum_completed_trades_enforced"] is False
    assert result["production_promotion_eligible"] is False


def test_genome_description_serializes_dynamic_exit_and_trailing_genes() -> None:
    genes = repair_individual([2, 2, 4, 0, 1, 2, 3, 1, 4, 2, 1])
    genome = _strategy_genome(genes)
    assert genome.timeframe.duration_seconds == 15 * 60
    assert genome.position_size_rule.constant == 0.5
    assert genome.stop_rule.operator == "activated_trailing_stop"
    assert len(genome_sha256(genome)) == 64
    assert "ema_trend_gap" in str(genome.exit_rule)
    assert EXIT_MODES[genes[10]] in EXIT_MODES


def test_resampled_feature_availability_uses_candle_close_for_temporal_split() -> None:
    count = 240
    open_times = np.arange(count, dtype=np.int64) * 60_000
    open_prices = 100.0 + np.arange(count, dtype=np.float64) * 0.001
    close_prices = open_prices + 0.01
    raw = pd.DataFrame(
        {
            "open_time_ms": open_times,
            "close_time_ms": open_times + 59_999,
            "open": open_prices,
            "high": close_prices + 0.01,
            "low": open_prices - 0.01,
            "close": close_prices,
            "volume": np.ones(count),
        }
    )
    bars = _resample(raw, 60)
    features = _build_features(bars)
    train_cutoff_ms = 150 * 60_000
    validation_end_ms = 240 * 60_000
    assert bars["open_time_ms"].iloc[2] < train_cutoff_ms
    assert bars["close_time_ms"].iloc[2] >= train_cutoff_ms
    assert features.timestamp[2] >= train_cutoff_ms
    train_indexes = np.flatnonzero(features.timestamp < train_cutoff_ms)
    validation_indexes = np.flatnonzero(
        (features.timestamp >= train_cutoff_ms)
        & (features.timestamp < validation_end_ms)
    )
    assert train_indexes.tolist() == [0, 1]
    assert validation_indexes.tolist() == [2, 3]


def test_required_timeframe_quota_includes_15m_and_is_unique() -> None:
    class FakeToolbox:
        def __init__(self) -> None:
            self.calls = 0

        def individual(self) -> list[int]:
            n = self.calls
            self.calls += 1
            return [0, 2, n % 8, 0, 1, n % 6, (n // 2) % 6, n % 4, n % 6, n % 5, n % 4]

    toolbox = FakeToolbox()
    population = [
        [
            TIMEFRAMES.index(1),
            2,
            i % 8,
            0,
            1,
            i % 6,
            (i // 2) % 6,
            i % 4,
            i % 6,
            i % 5,
            i % 4,
        ]
        for i in range(20)
    ]
    result = _ensure_timeframe_coverage(population, toolbox, cache={})
    assert len(result) == len(population)
    assert 15 in REQUIRED_TIMEFRAMES
    for timeframe in REQUIRED_TIMEFRAMES:
        gene = TIMEFRAMES.index(timeframe)
        assert (
            len({tuple(item) for item in result if item[0] == gene})
            >= MIN_REQUIRED_TIMEFRAME_POPULATION
        )


def test_population_and_generation_limits_fail_closed(tmp_path: Path) -> None:
    from scripts.run_deap_evolution import run_evolution

    with pytest.raises(ValueError, match="population_size_must_be_between"):
        run_evolution(
            tmp_path / "missing.csv", tmp_path / "missing.json", tmp_path, 99, 10
        )
    with pytest.raises(ValueError, match="generations_must_be_between"):
        run_evolution(
            tmp_path / "missing.csv", tmp_path / "missing.json", tmp_path, 128, 9
        )
    with pytest.raises(ValueError, match="generations_must_be_between"):
        run_evolution(
            tmp_path / "missing.csv", tmp_path / "missing.json", tmp_path, 128, 21
        )
