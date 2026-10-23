"""Versioned, fail-closed strategy selection; proposals only, never order execution."""

from services.strategy_selection.selector import select_strategy

__all__ = ["select_strategy"]
