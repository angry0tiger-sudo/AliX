# Novaryn Adapter: Open Task

**Status:** OPEN / BLOCKED — no source repository access has been established.

## Acceptance criteria
- Obtain the official repository URL from the owner.
- Pin an immutable commit SHA and verify its license and provenance.
- Review its public API and compatibility with AliX contracts.
- Implement an adapter behind the provider interface; no direct dependency from core logic.
- Add contract tests using mocks/fixtures only.
- Document supported data, error behavior, rate limits, and failure modes.
- Do not enable network access or live trading in CI as part of this task.

This directory intentionally contains no invented adapter implementation or assumed compatibility.
