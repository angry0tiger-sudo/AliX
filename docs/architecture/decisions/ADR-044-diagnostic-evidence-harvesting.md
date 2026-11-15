# ADR-044: Diagnostic Evidence Harvesting

- **Status:** Accepted
- **Date:** 2026-11-15 UTC
- **Branch:** research/nobitex-hf-strategy
- **Decision scope:** Evidence retention and diagnostic workflow architecture only
- **Related:** ADR-042, ADR-043, FOUR_ARM_SIGNAL_DENSITY_DIAGNOSTIC_SPEC.md
- **Frozen main:** 44bcaeae7eefd358cf063394b0e7f4f0b83af8eb

## Context

A temporary diagnostic-only implementation was used to decompose Run #10 signal density and entry attrition. It consumed only the immutable pre-hack confirmatory prefix and the retained Run #10 forecast cache. Three separate GitHub Actions executions produced the same extracted JSON and Markdown report bytes.

The diagnostic script and its workflow are absent from the current harvested branch tree. The diagnostic executable is not being restored. The output evidence, however, remains available in three unexpired GitHub artifacts and is now preserved as content-addressed repository records.

## Evidence

The three artifact records are:

| Run ID | Artifact ID | Producer HEAD | Archive SHA-256 |
|---:|---:|---|---|
| 43739166184 | 14171940837 | 89ce923338ba977392539c6896d6f138cad36151 | f12ce09667fa2768114f99e0fe21b95c4f5b792f16fce9393f3838635dfdf86a |
| 43739941516 | 14172285941 | 719b96dfffac265c4cf384236703d998a97d1326 | 4cfbd156c2d9f02f4e87448b682459b4c521dcab7a4dac1033933824a5b1decd |
| 43740274764 | 14171877977 | 0ca123db055b2ae28e2f9418982cd875f49d14a5 | b094f2f4643a9eed8d3e4770d241e17a39df10fe8cd9e53f4db3f5a6b26f8d28 |

Each ZIP is 4,843 bytes and contains exactly two files:
- diagnostic-report.json — 47,323 bytes, SHA-256 2d759005f54e4008bf504e046f4ba9b2cad397f2092de9887b12110751f4ce2c
- diagnostic-report.md — 3,404 bytes, SHA-256 8e89419737f8f6fea6222bdae09930f07a66cacd623c7c9ace77db3c6c81841e

Both extracted members are byte-identical across all three runs. The ZIP wrapper hashes differ, so only the extracted report payloads are asserted deterministic.

The harvested result in docs/research/diagnostics/FOUR_ARM_SIGNAL_DENSITY_DIAGNOSTIC_RESULTS.json and docs/research/diagnostics/FOUR_ARM_SIGNAL_DENSITY_DIAGNOSTIC_HARVEST_REPORT.md preserves:
- 9 B/C/D × threshold cells and 27 fold rows;
- 10,500 forecast-covered decision bars per cell;
- exact eligible-signal, entry-rejection, position-occupancy and completed-trade counts;
- all eight relative forecast-margin histogram bins per cell;
- explicit MISSING status for risk-gate rejects, because the Run #10 runner has no separate pre-entry risk gate;
- exact Run #10 manifest reconciliation and source/cache provenance.

The audit used 27,825 rows from the immutable pre-hack prefix with SHA-256 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e. It reused the 1,138-anchor Run #10 forecast cache (artifact 14163527266) and made zero new model-inference calls. It created no economic metric or trial, and did not access holdout.

## Decision

A diagnostic-only executable does not need permanent residence in the production/research source tree for its evidence to be admissible. A one-off, isolated workflow may be retired after execution if all of the following survive:
1. immutable workflow run, producer HEAD, and artifact ID are recorded;
2. archive digests are verified against the GitHub API and the extracted member hashes are verified;
3. structured aggregate output is independently reconciled to the source manifests/contracts;
4. data boundaries, cache identity, and safety invariants are explicit;
5. the resulting evidence is committed in a stable, append-only documentation/data path.

This is an evidence-retention rule, not permission to run arbitrary ephemeral code. Future diagnostic work still requires an isolated workflow, a reviewed data boundary, no economic-execution capability, aggregate-only outputs, and a source/evidence reconciliation gate. If the diagnostic is needed again, it must undergo a new review; this ADR does not authorize restoring or dispatching it.

## Consequences and safety

- The diagnostic script/workflow remain absent; this commit adds only JSON/Markdown evidence and ADR/memo appendices.
- ADR-042 and all twelve Run #10 trial records remain unchanged.
- ADR-043 remains prospective only; Run #10 is not retroactively FUTILE.
- GLOBAL_N=UNKNOWN; DSR numeric sensitivities remain MISSING.
- main remains frozen at 44bcaeae7eefd358cf063394b0e7f4f0b83af8eb.
- Execution lock remains NOT_AUTHORIZED; EXECUTION_TRIGGER and ACQUISITION_TRIGGER remain PROBE.
- No holdout access, threshold change, inference, economic trial, private API call or order attempt is authorized.

Architecture First. Evidence Always. Fail Closed.
