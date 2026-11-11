"""Command entry point for AI-2 Revisited DEAP rule discovery."""

import importlib
from typing import Any


def main() -> None:
    engine: Any = importlib.import_module("research.deap.engine")
    engine.run_cli()


if __name__ == "__main__":
    main()
