"""Architecture assertions for the strategy-selection capability boundary."""

from __future__ import annotations

import ast
from pathlib import Path


def test_selector_modules_do_not_import_execution_or_risk_capabilities() -> None:
    source_paths = (
        Path("services/strategy_selection/__init__.py"),
        Path("services/strategy_selection/contracts.py"),
        Path("services/strategy_selection/registry.py"),
        Path("services/strategy_selection/selector.py"),
    )
    forbidden_roots = {
        "ccxt",
        "control_plane",
        "freqtrade",
        "integrations",
        "risk",
        "requests",
        "httpx",
        "socket",
        "subprocess",
    }
    imports: list[str] = []
    for source_path in source_paths:
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imports.extend(
                    alias.name.split(".", maxsplit=1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module.split(".", maxsplit=1)[0])

    assert not forbidden_roots.intersection(imports)


def test_selector_does_not_enable_existing_freqtrade_strategy_signals() -> None:
    strategy_source = Path("user_data/strategies/AliXStrategy.py").read_text(
        encoding="utf-8"
    )
    assert 'dataframe["enter_long"] = 0' in strategy_source
    assert 'dataframe["exit_long"] = 0' in strategy_source
