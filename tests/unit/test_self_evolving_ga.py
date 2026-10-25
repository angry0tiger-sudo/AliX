from __future__ import annotations

import numpy as np
import pytest

from scripts.run_deap_evolution import (
    DIRECTIONAL_EXIT,
    EMA_ENTRY,
    EvaluationMetrics,
    _strategy_genome,
    _thresholds,
    build_position_states,
    fitness_score,
    metrics_from_positions,
    repair_individual,
)
from services.ai_core.self_evolving_genome import genome_sha256


def test_repair_keeps_genes_in_range_and_ema_periods_ordered() -> None:
    genes = [99, 1, 7, 5, 0, 4, 5, 3, 5]
    repaired = repair_individual(genes)
    assert tuple(repaired[:2]) == (5, 1)
    assert repaired[2] == 0
    from scripts.run_deap_evolution import FASTS, SLOWS

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
    assert -0.011 < metrics.net_expectancy < -0.008
    assert metrics.win_rate == 0.0
    assert metrics.net_return < 0.0


def test_fitness_penalizes_drawdown_churn_and_small_samples() -> None:
    baseline = EvaluationMetrics(40, 0.55, 0.002, 0.05, 0.10)
    drawdown = EvaluationMetrics(40, 0.55, 0.002, 0.05, 0.40)
    churn = EvaluationMetrics(450, 0.55, 0.002, 0.05, 0.10)
    small = EvaluationMetrics(5, 0.80, 0.002, 0.05, 0.10)
    assert fitness_score(baseline) > fitness_score(drawdown)
    assert fitness_score(baseline) > fitness_score(churn)
    assert fitness_score(baseline) > fitness_score(small)


def test_genome_description_is_serializable_and_hashable() -> None:
    genes = repair_individual([2, 2, 4, 0, 1, 2, 3, 1, 4])
    genome = _strategy_genome(genes)
    assert genome.timeframe.duration_seconds == 15 * 60
    assert genome.position_size_rule.constant == 0.5
    assert genome.stop_rule.operator == "less_than"
    assert len(genome_sha256(genome)) == 64


def test_population_and_generation_limits_fail_closed(tmp_path) -> None:
    from scripts.run_deap_evolution import run_evolution

    with pytest.raises(ValueError, match="population_size_must_be_between"):
        run_evolution(
            tmp_path / "missing.csv", tmp_path / "missing.json", tmp_path, 99, 6
        )
    with pytest.raises(ValueError, match="generations_must_be_between"):
        run_evolution(
            tmp_path / "missing.csv", tmp_path / "missing.json", tmp_path, 128, 11
        )
