"""Versioned, serializable strategy-genome types for later evolutionary search.

The genome describes candidates. This module never executes generated source:
operators must be resolved by a separately tested primitive registry.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Literal

TimeframeUnit = Literal["minute", "hour", "day", "week", "month"]
_UNIT_SECONDS: dict[str, int] = {
    "minute": 60,
    "hour": 3_600,
    "day": 86_400,
    "week": 604_800,
    "month": 2_629_746,
}
_MAX_TIMEFRAME_SECONDS = 31 * 86_400
_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,63}\Z")
_RAW_FEATURES = frozenset({"open", "high", "low", "close", "volume"})
_MAX_EXPRESSION_DEPTH = 16
_MAX_EXPRESSION_NODES = 256


@dataclass(frozen=True, slots=True)
class TimeframeGene:
    """A timeframe encoded as a duration unit, not a preferred fixed shortlist."""

    amount: int
    unit: TimeframeUnit

    def __post_init__(self) -> None:
        if isinstance(self.amount, bool) or not isinstance(self.amount, int):
            raise ValueError("timeframe_amount_must_be_integer")
        if self.unit not in _UNIT_SECONDS:
            raise ValueError("timeframe_unit_unsupported")
        duration = self.amount * _UNIT_SECONDS[self.unit]
        if self.amount <= 0 or not 60 <= duration <= _MAX_TIMEFRAME_SECONDS:
            raise ValueError("timeframe_out_of_search_range")

    @property
    def duration_seconds(self) -> int:
        return self.amount * _UNIT_SECONDS[self.unit]


@dataclass(frozen=True, slots=True)
class IndicatorGene:
    """An operator name, finite parameters, and its own source timeframe."""

    key: str
    operator: str
    parameters: tuple[float, ...]
    timeframe: TimeframeGene

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.fullmatch(self.key):
            raise ValueError("indicator_key_invalid")
        if not _NAME_PATTERN.fullmatch(self.operator):
            raise ValueError("indicator_operator_invalid")
        for value in self.parameters:
            _require_finite(value, "indicator_parameter")


@dataclass(frozen=True, slots=True)
class ExpressionGene:
    """A safe expression node; evaluation is deferred to a primitive allowlist."""

    operator: str
    children: tuple[ExpressionGene, ...] = ()
    feature: str | None = None
    constant: float | None = None
    parameters: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _validate_expression(self)


@dataclass(frozen=True, slots=True)
class StrategyGenome:
    """Complete candidate genes for timing, features, entries, exits and sizing."""

    genome_id: str
    timeframe: TimeframeGene
    indicators: tuple[IndicatorGene, ...]
    entry_rule: ExpressionGene
    exit_rule: ExpressionGene
    position_size_rule: ExpressionGene
    stop_rule: ExpressionGene

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.fullmatch(self.genome_id):
            raise ValueError("genome_id_invalid")
        keys = [indicator.key for indicator in self.indicators]
        if len(keys) != len(set(keys)):
            raise ValueError("indicator_keys_must_be_unique")
        available_features = _RAW_FEATURES.union(keys)
        for expression in (
            self.entry_rule,
            self.exit_rule,
            self.position_size_rule,
            self.stop_rule,
        ):
            for feature in _expression_features(expression):
                if feature not in available_features:
                    raise ValueError("genome_references_unknown_feature")


def _require_finite(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field}_must_be_numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field}_must_be_finite")


def _validate_expression(root: ExpressionGene) -> None:
    count = 0

    def visit(node: ExpressionGene, depth: int) -> None:
        nonlocal count
        count += 1
        if count > _MAX_EXPRESSION_NODES:
            raise ValueError("expression_node_budget_exceeded")
        if depth > _MAX_EXPRESSION_DEPTH:
            raise ValueError("expression_depth_budget_exceeded")
        if not _NAME_PATTERN.fullmatch(node.operator):
            raise ValueError("expression_operator_invalid")
        for value in node.parameters:
            _require_finite(value, "expression_parameter")
        if node.operator == "feature":
            if (
                node.feature is None
                or not _NAME_PATTERN.fullmatch(node.feature)
                or node.constant is not None
                or node.children
                or node.parameters
            ):
                raise ValueError("feature_expression_invalid")
        elif node.operator == "constant":
            if (
                node.constant is None
                or node.feature is not None
                or node.children
                or node.parameters
            ):
                raise ValueError("constant_expression_invalid")
            _require_finite(node.constant, "expression_constant")
        elif node.feature is not None or node.constant is not None or not node.children:
            raise ValueError("operator_expression_invalid")
        for child in node.children:
            visit(child, depth + 1)

    visit(root, 1)


def _expression_features(root: ExpressionGene) -> set[str]:
    features: set[str] = set()
    nodes = [root]
    while nodes:
        node = nodes.pop()
        if node.operator == "feature" and node.feature is not None:
            features.add(node.feature)
        nodes.extend(node.children)
    return features


def serialize_genome(genome: StrategyGenome) -> str:
    """Return stable JSON suitable for an immutable candidate archive."""

    return json.dumps(asdict(genome), sort_keys=True, separators=(",", ":"))


def genome_sha256(genome: StrategyGenome) -> str:
    """Content address for reproducible candidate lineage and learning ledgers."""

    return hashlib.sha256(serialize_genome(genome).encode("utf-8")).hexdigest()
