"""The four-arm runner must remain importable without model-runtime packages."""

from __future__ import annotations

import ast
from pathlib import Path


def test_runner_has_no_module_level_model_runtime_imports() -> None:
    source = Path("scripts/run_four_arm_experiment.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    forbidden = {"torch", "chronos", "model", "huggingface_hub"}
    imported: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(forbidden)
