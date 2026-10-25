from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.run_deap_evolution import (
    DIRECTIONAL_EXIT,
    EMA_ENTRY,
    FASTS,
    MAX_DRAWDOWN,
    MIN_REQUIRED_TIMEFRAME_POPULATION,
    REQUIRED_TIMEFRAMES,
    SLOWS,
    CandidateEvaluation,
    EvaluationMetrics,
    TIMEFRAMES,
    _economic_assessment,
    _ensure_timeframe_coverage,
    _select_by_eligibility,
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
) -> EvaluationMetrics:
    return EvaluationMetrics(
        trades, win_rate, expectancy if trades else None, net_return, drawdown
    )


def _candidate(
    training: EvaluationMetrics, validation: EvaluationMetrics
) -> CandidateEvaluation:
    reasons = []
    if training.trade_count < 30:
        reasons.append("training_trade_count_below_minimum")
    if training.net_expectancy is None:
        reasons.append("training_expectancy_unavailable")
    if validation.trade_count < 10:
        reasons.append("validation_trade_count_below_minimum")
    if training.max_drawdown > MAX_DRAWDOWN:
        reasons.append("training_drawdown_above_limit")
    return CandidateEvaluation(
        training=training,
        validation=validation,
        fitness=fitness_score(training, validation),
        eligibility_reasons=tuple(reasons),
    )


def test_repair_keeps_genes_in_range_and_ema_periods_ordered() -> None:
    genes = [99, 1, 7, 5, 0, 4, 5, 3, 5]
    repaired = repair_individual(genes)
    assert tuple(repaired[:2]) == (len(TIMEFRAMES) - 1, 1)
    assert repaired[2] == 0
    assert FASTS[repaired[3]] < SLOWS[repaired[4]]


def test_repair_canonicalizes_inactive_ema_genes_for_other_indicators() -> None:
    genes = [0, 0, 7, 5, 7, 1, 2, 0, 1]
    repaired = repair_individual(genes)
    assert repaired[2] == 7
    assert repaired[3:5] == [0, 1]


def test_directional_thresholds_do_not_overlap_for_entry_and_exit() -> None:
    for entry_index in range(len(EMA_ENTRY)):
        for exit_index in range(len(DIRECTIONAL_EXIT)):
            thresholds = _thresholds([0, 1, 0, 0, 1, entry_index, exit_index, 0, 0])
            assert thresholds[1] < thresholds[0]


def test_close_based_trailing_stop_exits_only_after_causal_close() -> None:
    close = np.array([100.0, 101.0, 110.0, 106.0, 104.0, 105.0, 108.0])
    entry = np.array([False, True, True, True, True, True, True])
    exit_condition = np.zeros(7, dtype=bool)
    positions = build_position_states(entry, exit_condition, close, 0.05)
    assert positions.tolist() == [False, True, True, True, False, False, False]


def test_net_expectancy_includes_both_sides_fee_and_slippage() -> None:
    opens = np.full(5, 100.0)
    closes = np.full(5, 100.0)
    positions = np.array([True, True, False, False, False])
    metrics = metrics_from_positions(opens, closes, positions, 0.5)
    assert metrics.trade_count == 1
    assert metrics.net_expectancy is not None
    assert -0.011 < metrics.net_expectancy < -0.008
    assert metrics.win_rate == 0.0
    assert metrics.net_return < 0.0


def test_max_drawdown_uses_costed_portfolio_equity_curve() -> None:
    opens = np.array([100.0, 100.0, 200.0, 100.0, 100.0])
    closes = np.array([100.0, 200.0, 100.0, 100.0, 100.0])
    positions = np.array([True, True, False, False, False])
    metrics = metrics_from_positions(opens, closes, positions, 1.0)
    assert metrics.max_drawdown > 0.15
    assert metrics.net_return < 0.0


def test_no_completed_trades_have_unavailable_expectancy_and_zero_fitness() -> None:
    opens = np.full(8, 100.0)
    closes = np.full(8, 100.0)
    positions = np.zeros(8, dtype=bool)
    no_trades = metrics_from_positions(opens, closes, positions, 0.25)
    assert no_trades.trade_count == 0
    assert no_trades.net_expectancy is None
    assert no_trades.win_rate == 0.0
    assert fitness_score(no_trades, _metrics(10)) == 0.0


@pytest.mark.parametrize(
    "training_trades,validation_trades",
    [(29, 10), (30, 9), (0, 0)],
)
def test_sample_gate_returns_zero_fitness(
    training_trades: int, validation_trades: int
) -> None:
    training = _metrics(training_trades)
    validation = _metrics(validation_trades)
    assert fitness_score(training, validation) == 0.0
    result = _candidate(training, validation)
    assert not result.sample_eligible
    assert result.fitness == 0.0


def test_30_training_and_10_validation_trades_pass_sample_gate() -> None:
    training = _metrics(30, expectancy=-0.001)
    validation = _metrics(10, expectancy=0.001)
    result = _candidate(training, validation)
    assert result.sample_eligible
    assert result.fitness == pytest.approx(-0.001)


def test_zero_fitness_sample_invalid_cannot_beat_negative_valid_candidate() -> None:
    valid_individual = [TIMEFRAMES.index(60), 2, 0, 0, 1, 0, 0, 0, 0]
    invalid_individual = [TIMEFRAMES.index(240), 2, 1, 0, 1, 1, 1, 1, 1]
    valid = _candidate(_metrics(35, expectancy=-0.01), _metrics(12))
    invalid = _candidate(_metrics(29), _metrics(12))
    assert valid.fitness < invalid.fitness
    assert invalid.fitness == 0.0
    cache = {tuple(valid_individual): valid, tuple(invalid_individual): invalid}
    selected = _select_by_eligibility(
        [invalid_individual, valid_individual], 1, cache=cache, tournsize=2
    )
    assert tuple(selected[0]) == tuple(valid_individual)


def test_drawdown_over_15_percent_gets_severe_non_positive_fitness() -> None:
    low_risk = _metrics(40, expectancy=0.02, drawdown=0.10)
    extreme_return_high_risk = _metrics(40, expectancy=0.50, drawdown=0.20)
    score_low_risk = fitness_score(low_risk, _metrics(10))
    score_high_risk = fitness_score(extreme_return_high_risk, _metrics(10))
    assert score_low_risk > 0.0
    assert score_high_risk < 0.0
    assert score_high_risk < score_low_risk
    assert not _candidate(extreme_return_high_risk, _metrics(10)).selection_eligible


def test_economic_assessment_rejects_low_sample_or_unvalidated_candidate() -> None:
    training = _metrics(29, expectancy=0.12, net_return=0.03, drawdown=0.01)
    validation = _metrics(9, expectancy=None, net_return=0.0, drawdown=0.0)
    result = _economic_assessment(training, validation)
    assert result["status"] == "NO-GO"
    assert "training_trade_count_below_minimum" in result["reasons"]
    assert "validation_trade_count_below_minimum" in result["reasons"]
    assert result["final_holdout_used"] is False
    assert result["production_promotion_eligible"] is False


def test_no_candidate_assessment_fails_closed() -> None:
    result = _economic_assessment(None, None)
    assert result["status"] == "NO-GO"
    assert (
        "no_candidate_meets_training_validation_sample_and_training_drawdown_gates"
        in result["reasons"]
    )
    assert result["final_holdout_used"] is False
    assert result["production_promotion_eligible"] is False


def test_fitness_penalizes_churn_and_invalid_samples() -> None:
    baseline = _metrics(40, expectancy=0.002, drawdown=0.10)
    churn = _metrics(450, expectancy=0.002, drawdown=0.10)
    small = _metrics(5, expectancy=0.80, drawdown=0.10)
    validation = _metrics(10)
    assert fitness_score(baseline, validation) > fitness_score(churn, validation)
    assert fitness_score(baseline, validation) > fitness_score(small, validation)


def test_genome_description_is_serializable_and_hashable() -> None:
    genes = repair_individual([2, 2, 4, 0, 1, 2, 3, 1, 4])
    genome = _strategy_genome(genes)
    assert genome.timeframe.duration_seconds == 15 * 60
    assert genome.position_size_rule.constant == 0.5
    assert genome.stop_rule.operator == "less_than"
    assert len(genome_sha256(genome)) == 64


def test_required_timeframe_quota_is_preserved_in_population() -> None:
    class FakeToolbox:
        def __init__(self) -> None:
            self.calls = 0

        def individual(self) -> list[int]:
            n = self.calls
            self.calls += 1
            return [0, 2, n % 8, 0, 1, n % 6, (n // 2) % 6, n % 4, n % 6]

    toolbox = FakeToolbox()
    population = [
        [TIMEFRAMES.index(1), 2, i % 8, 0, 1, i % 6, (i // 2) % 6, i % 4, i % 6]
        for i in range(20)
    ]
    result = _ensure_timeframe_coverage(population, toolbox, cache={})
    assert len(result) == len(population)
    for timeframe in REQUIRED_TIMEFRAMES:
        gene = TIMEFRAMES.index(timeframe)
        assert (
            sum(1 for item in result if item[0] == gene)
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
