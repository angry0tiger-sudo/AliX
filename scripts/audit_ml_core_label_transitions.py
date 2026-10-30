"""CLI entry point for the offline V4-C label-transition audit."""

from services.ml_core.label_transition_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
