from __future__ import annotations

import math

import pytest

from services.ai_core.self_evolving_genome import (
    ExpressionGene,
    IndicatorGene,
    StrategyGenome,
    TimeframeGene,
    genome_sha256,
    serialize_genome,
)


def _feature(name: str) -> ExpressionGene:
    return ExpressionGene(operator="feature", feature=name)


def _constant(value: float) -> ExpressionGene:
    return ExpressionGene(operator="constant", constant=value)


def _genome() -> StrategyGenome:
    timeframe = TimeframeGene(amount=1, unit="minute")
    indicator = IndicatorGene(
        key="rsi_fast",
        operator="rsi",
        parameters=(14.0,),
        timeframe=timeframe,
    )
    entry = ExpressionGene(
        operator="greater_than",
        children=(_feature("rsi_fast"), _constant(50.0)),
    )
    exit_rule = ExpressionGene(
        operator="less_than",
        children=(_feature("rsi_fast"), _constant(45.0)),
    )
    return StrategyGenome(
        genome_id="candidate_001",
        timeframe=timeframe,
        indicators=(indicator,),
        entry_rule=entry,
        exit_rule=exit_rule,
        position_size_rule=_constant(0.5),
        stop_rule=_constant(0.03),
    )


def test_genome_contains_timeframe_features_entries_exits_sizing_and_stop() -> None:
    genome = _genome()
    assert genome.timeframe.duration_seconds == 60
    assert genome.indicators[0].operator == "rsi"
    assert genome.entry_rule.operator == "greater_than"
    assert genome.exit_rule.operator == "less_than"
    assert genome.position_size_rule.constant == 0.5
    assert genome.stop_rule.constant == 0.03


def test_genome_serialization_is_stable_and_content_addressed() -> None:
    genome = _genome()
    serialized = serialize_genome(genome)
    assert serialized == serialize_genome(genome)
    assert len(genome_sha256(genome)) == 64
    assert genome_sha256(genome) == genome_sha256(_genome())


@pytest.mark.parametrize(
    "amount, unit",
    [
        (0, "minute"),
        (1, "second"),
        (100_000, "minute"),
        (True, "minute"),
    ],
)
def test_invalid_timeframe_genes_fail_closed(amount: int, unit: str) -> None:
    with pytest.raises(ValueError):
        TimeframeGene(amount=amount, unit=unit)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "key": "9bad",
            "operator": "rsi",
            "parameters": (),
            "timeframe": TimeframeGene(1, "minute"),
        },
        {
            "key": "valid",
            "operator": "exec()",
            "parameters": (),
            "timeframe": TimeframeGene(1, "minute"),
        },
        {
            "key": "valid",
            "operator": "rsi",
            "parameters": (math.nan,),
            "timeframe": TimeframeGene(1, "minute"),
        },
    ],
)
def test_invalid_indicator_genes_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        IndicatorGene(**kwargs)  # type: ignore[arg-type]


def test_invalid_expression_nodes_fail_closed() -> None:
    with pytest.raises(ValueError, match="feature_expression_invalid"):
        ExpressionGene(operator="feature")
    with pytest.raises(ValueError, match="expression_constant_must_be_finite"):
        ExpressionGene(operator="constant", constant=math.inf)
    with pytest.raises(ValueError, match="operator_expression_invalid"):
        ExpressionGene(operator="add")
    with pytest.raises(ValueError, match="expression_operator_invalid"):
        ExpressionGene(operator="exec()")


def test_expression_depth_budget_is_enforced() -> None:
    node = _feature("close")
    with pytest.raises(ValueError, match="expression_depth_budget_exceeded"):
        for _ in range(17):
            node = ExpressionGene(operator="positive", children=(node,))


def test_genome_rejects_duplicate_or_unknown_indicator_features() -> None:
    genome = _genome()
    duplicate = IndicatorGene(
        key="rsi_fast",
        operator="ema",
        parameters=(8.0,),
        timeframe=TimeframeGene(1, "minute"),
    )
    with pytest.raises(ValueError, match="indicator_keys_must_be_unique"):
        StrategyGenome(
            genome_id="duplicate",
            timeframe=genome.timeframe,
            indicators=(genome.indicators[0], duplicate),
            entry_rule=genome.entry_rule,
            exit_rule=genome.exit_rule,
            position_size_rule=genome.position_size_rule,
            stop_rule=genome.stop_rule,
        )
    with pytest.raises(ValueError, match="unknown_feature"):
        StrategyGenome(
            genome_id="unknown",
            timeframe=genome.timeframe,
            indicators=genome.indicators,
            entry_rule=_feature("future_data"),
            exit_rule=genome.exit_rule,
            position_size_rule=genome.position_size_rule,
            stop_rule=genome.stop_rule,
        )
