"""Regression tests for the one-time 32-hex execution nonce guard."""

from __future__ import annotations

from pathlib import Path
import subprocess

WORKFLOW_PATH = Path(".github/workflows/four-arm-real-experiment.yml")
TRIGGER_PATH = "config/four_arm/EXECUTION_TRIGGER"


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit_trigger(repository: Path, value: str, message: str) -> None:
    (repository / TRIGGER_PATH).write_text(f"{value}\n", encoding="utf-8")
    _git(repository, "add", TRIGGER_PATH)
    _git(repository, "commit", "-m", message)


def test_workflow_guard_validates_and_searches_the_exact_nonce() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert '[[ ! "$trigger_value" =~ ^[0-9a-f]{32}$ ]]' in workflow
    assert 'git log "$EVENT_BEFORE" -S"$trigger_value"' in workflow
    assert '-S"EXECUTE"' not in workflow


def test_git_history_accepts_a_fresh_nonce_and_rejects_reuse(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "AliX Guard Test")
    _git(repository, "config", "user.email", "alix-guard-test@example.invalid")

    _commit_trigger(repository, "PROBE", "initialize execution trigger")
    before_first_use = _git(repository, "rev-parse", "HEAD")
    nonce = "0123456789abcdef0123456789abcdef"
    _commit_trigger(repository, nonce, "first one-time nonce use")

    fresh_history = _git(
        repository,
        "log",
        before_first_use,
        f"-S{nonce}",
        "--format=%H",
        "--",
        TRIGGER_PATH,
    )
    assert fresh_history == ""

    _commit_trigger(repository, "PROBE", "revoke first nonce")
    before_reuse = _git(repository, "rev-parse", "HEAD")
    _commit_trigger(repository, nonce, "attempt to reuse nonce")

    replay_history = _git(
        repository,
        "log",
        before_reuse,
        f"-S{nonce}",
        "--format=%H",
        "--",
        TRIGGER_PATH,
    )
    assert replay_history
