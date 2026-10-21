# AliX Project Rules

> دستورات مالک همواره بالاتر از اسناد است، تحویل فازها باید End-to-End باشد و هیچ فازی نباید نیمه‌کاره رها شود.

## Authority and change control
- The project owner is the final authority for product and architecture decisions. Record material directives in `docs/governance/owner-directives/` with impact, affected documents, tests, and resulting commit.
- This document does not override platform safety, law, or security controls.
- Do not silently reinterpret a direct owner instruction. If it conflicts with a safety boundary or an immutable technical constraint, document the exact conflict and the smallest safe alternative.

## Delivery gates
- A phase is complete only when its acceptance criteria are documented, implemented, tested, and evidenced by a green CI run on the exact candidate commit.
- No partial phase is to be represented as complete. Open blockers must remain visible in the phase report.
- Never claim a test or workflow passed without reading its actual result.

## Trading and external services
- All CI, development, integration, and demonstration tests MUST use mocks, fixtures, backtests, or paper/dry-run mode.
- No live orders, real exchange credentials, or live external-service writes are permitted from GitHub Actions.
- Never commit tokens, passwords, private keys, account identifiers, or real .env files.
- Default exchange connectivity and order execution must fail closed when configuration, market metadata, risk approval, or data quality is missing.
- Do not invent exchange market pairs or claim the Novaryn adapter is compatible until its canonical URL, pinned commit, license, and contract tests are verified.

## Dependency and data governance
- Pin direct dependencies and record the Python/runtime compatibility matrix. Revalidate licenses and upstream compatibility before release.
- Register every external data provider, model, dataset, and API with license/terms, rate limits, retention, and fallback behavior.
- Model outputs are proposals, never risk approvals. Hard risk limits and kill switches remain independent of model confidence.

## Review evidence
- Each phase report records commit SHA, acceptance criteria, CI run URL/status, test evidence, known limitations, dependency/model hashes where relevant, and explicit GO/NO-GO.
