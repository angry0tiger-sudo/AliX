## Summary

Describe the change and its expected behavior.

## Phase / related work

- Phase:
- Related issue or acceptance criterion:
- Architectural or contract references:

## Change classification

- [ ] Documentation only
- [ ] Adapter/transport contract
- [ ] Dependency or runtime version
- [ ] Security, permissions, or credentials
- [ ] Data/model/provider contract
- [ ] Risk, execution, or reconciliation behavior

## Validation evidence

- [ ] Python compile and Ruff lint/format
- [ ] Strict Mypy
- [ ] Bandit and dependency audit
- [ ] Architecture and complexity gates
- [ ] Unit/contract tests and coverage threshold
- [ ] Starter Kit/documentation validation
- [ ] Freqtrade runtime smoke (when applicable)
- CI run URL and tested commit SHA:

## Safety and regression checks

- [ ] No API key, token, private key, or account secret added to source, fixtures, logs, or artifacts.
- [ ] Exchange, notification, and network behavior is mocked in CI; no real external side effect is triggered.
- [ ] No live order creation/cancellation or blind retry after unknown order outcomes.
- [ ] Freqtrade remains dry-run/stopped, zero-open-trade, and without approved markets unless a separately authorized release explicitly changes this.
- [ ] No Freqtrade core/site-packages patch or second order lifecycle was introduced.
- [ ] Source/data/model license and provenance impacts are documented.

## External deployment gates

List every unverified live/environment-specific requirement. A green mock test is not evidence of live compatibility.

## Rollback

Describe how the change can be reverted and any data/configuration implications.

## Review and decision

- [ ] Tests and evidence reviewed
- [ ] Security/contract implications reviewed
- [ ] Owner/reviewer decision recorded
- Merge decision: GO / NO-GO / CONDITIONAL GO
