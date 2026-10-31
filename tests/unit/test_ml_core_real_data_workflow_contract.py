from __future__ import annotations

from pathlib import Path


def test_v4e_workflow_shares_barrier_multipliers_across_steps() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "ml-core-real-data-baseline.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")
    job = workflow.split("  diagnostic-classification:\\n", maxsplit=1)[1]
    job_environment = job.split("    steps:\\n", maxsplit=1)[0]
    training_step = job.split(
        "      - name: Train V4-E hierarchical 12-candle diagnostics\\n",
        maxsplit=1,
    )[1].split(
        "      - name: Validate V4-E hierarchical report and safety invariants\\n",
        maxsplit=1,
    )[0]
    validation_step = job.split(
        "      - name: Validate V4-E hierarchical report and safety invariants\\n",
        maxsplit=1,
    )[1].split(
        "      - name: Verify GitHub Step Summary contains V4-E metrics\\n",
        maxsplit=1,
    )[0]
    summary_step = job.split(
        "      - name: Verify GitHub Step Summary contains V4-E metrics\\n",
        maxsplit=1,
    )[1].split(
        "      - name: Upload V4-E hierarchical diagnostic report\\n",
        maxsplit=1,
    )[0]

    expected_environment = {
        "ALIX_TAKE_PROFIT_MULTIPLIER": (
            "${{ inputs.take_profit_multiplier == '' && '2.0' || inputs.take_profit_multiplier }}"
        ),
        "ALIX_STOP_LOSS_MULTIPLIER": (
            "${{ inputs.stop_loss_multiplier == '' && '1.0' || inputs.stop_loss_multiplier }}"
        ),
    }
    training_header = training_step.split("        run: |", maxsplit=1)[0]
    validation_header = validation_step.split("        run: |", maxsplit=1)[0]

    for variable, expression in expected_environment.items():
        assert f"      {variable}: {expression}\\n" in job_environment
        assert f"${variable}" in training_step
        assert f'os.environ["{variable}"]' in validation_step

    assert "        env:\\n" not in training_header
    assert "        env:\\n" not in validation_header
    assert 'echo "ALIX_TRAIN_STEP_SUMMARY=$GITHUB_STEP_SUMMARY" >> "$GITHUB_ENV"' in training_step
    assert 'test -n "$ALIX_TRAIN_STEP_SUMMARY"' in summary_step
    assert 'test -s "$ALIX_TRAIN_STEP_SUMMARY"' in summary_step
    assert '"$ALIX_TRAIN_STEP_SUMMARY"' in summary_step
    assert "test -s \"$GITHUB_STEP_SUMMARY\"" not in summary_step
