# Phase 5 Acceptance: Triple-Barrier and Meta-Labeling

Status: **IMPLEMENTATION CI PASS ON EXACT CODE SHA `59ab355a04c9b9e6452bb4767b81d4c0fab7f3a2`; OWNER REVIEW REQUIRED; ECONOMIC ACCEPTANCE NO-GO; NO TRADING**.

Canonical MRD: [`docs/knowledge_base/AliX_MRD_Dynamic_Risk_FA.md`](AliX_MRD_Dynamic_Risk_FA.md), section 4.6 Plan B.
Canonical Roadmap: [`docs/knowledge_base/AliX_Roadmap_Dynamic_Risk_FA.md`](AliX_Roadmap_Dynamic_Risk_FA.md), Phase 5.
Owner directive: [Phase 5 owner directive](../governance/owner-directives/2026-10-22-phase-5.md).
Architecture decision: [ADR-0005 Plan B](../architecture/decisions/ADR-0005-phase-5-plan-b.md).
Tracking issue: [#5](https://github.com/angry0tiger-sudo/AliX/issues/5).
Review PR: [#6](https://github.com/angry0tiger-sudo/AliX/pull/6).

## Exact-head CI evidence

- Branch: `feat/phase-5-triple-barrier-meta-labeling`.
- Tested code SHA: `59ab355a04c9b9e6452bb4767b81d4c0fab7f3a2`.
- [GitHub Actions run #369 / ID 39729707614](https://github.com/angry0tiger-sudo/AliX/actions/runs/39729707614) completed successfully on that exact SHA.
- `quality`: **success**. Compile, Ruff lint/format, strict Mypy, Bandit, dependency resolver/lock verification, `pip-audit`, import-linter, Xenon, unit/coverage, Phase 1 safety contracts and project-doc validation passed. Unit tests: **111 passed, 13 warnings**; coverage **91.38%** against 90% required; Phase 1 safety/dependency contracts: **4 passed**. `pip-audit` reported no known vulnerabilities.
- `freqtrade-runtime-smoke`: **success**. Freqtrade `2026.7`, existing model/strategy load and closed-config smoke passed. Phase 4 synthetic smoke and Phase 5 synthetic pipeline both ran; CI asserted the report commit SHA matched the PR head, the shared Data Quality Gate was VALID, OOF/meta checks passed, Champion stayed null, and the synthetic artifact upload succeeded. No exchange connection was requested.
- The tested code SHA above is the source for the following artifact; documentation updates in this review PR trigger their own CI run on the updated tip. Use the latest checks on [PR #6](https://github.com/angry0tiger-sudo/AliX/pull/6) for the final documentation commit.

## Reviewed artifact (SYNTHETIC / MECHANICS ONLY)

- Artifact: [`phase5-synthetic-meta-labeling-report`, ID 12404705949](https://github.com/angry0tiger-sudo/AliX/actions/runs/39729707614/artifacts/12404705949).
- ZIP size: 19,118 bytes; SHA-256: `18c35b30d49db09e41d15e360c5758cf0c70fd5e0e56967b97fb6429c83846b7`.
- JSON file: `alix-phase5-synthetic.json`; size 141,394 bytes; SHA-256: `3b8066bd0cc03b1a6daad816473d5b5461b4fd587a024570736ba9010d40464e`.
- Provenance: report `commit_sha=59ab355a04c9b9e6452bb4767b81d4c0fab7f3a2`; report schema `alix.phase5.synthetic-meta-labeling.v1`; status `SYNTHETIC_MECHANICS_ONLY`.
- Dataset: generated `SYNTHETIC/USDT`, 1h timeframe, 1,400 rows from `2020-01-01T00:00:00+00:00` through `2020-02-28T07:00:00+00:00`; 1,387 rows passed feature warm-up. Seed: 20261022. Dataset SHA-256: `aa0009d5ba495c1f69ad0c5cc9331066b2c6da1f00310875ace94f34a48a3e08`.
- Shared Phase 3 Data Quality Gate: **VALID**, passed `true`, empty reason codes, fingerprint `93e139ec09556c71cb77d40d90055f92f904f74ad9fd0bdf63b466d8e517df1f`. CI also contains a regression test that rejects an OHLCV gap through the same public gate.
- Experiment configuration SHA-256: `7855cc2be4558e5b3562fab9859a7246c47c2cf650713e8c633fa605007572f4`. Runtime versions: Python 3.12.15, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.0.
- Contracts: Triple-Barrier `alix.meta_labeling.triple-barrier.v1`; feature `alix.freqai.features.v1`; primary target `alix.freqai.forward-return.v1`; meta target semantics are `net_return_after_fee_and_slippage_strictly_positive`. The report states future labels are not features and meta training signals are `OUT_OF_FOLD_ONLY`.
- Primary OOF pipeline: 4 chronological folds; 800 held-out primary predictions; minimum primary train count 386; purge rule `primary label_end_index < test_start`.
- Triple-Barrier outcomes: 800 events, 451 stop-loss and 349 take-profit outcomes, zero vertical expiries in this synthetic realization. Barrier labels/outcomes remain separate from the feature matrix. The barrier config uses PT/SL multipliers 1.0, maximum holding 12 candles, fees 10 bps per side and slippage 5 bps per side. These are test assumptions, not live cost estimates.
- Meta-label distribution: 320 positive labels and 480 negative labels after applying the synthetic gross return minus modeled round-trip fees/slippage. The 0.5 research probability cutoff is a mechanics setting and is **not owner-approved economic policy**.
- Meta evaluation: four later-fold slots; three evaluated (600 OOS samples total) and one skipped because no prior OOF labels had matured. The maturity rule is `label_end_index < next_test_start`. Native JSON model reload passed for all three evaluated folds with maximum probability difference 0.0.
- The historical-rate baseline **outperformed the Meta-Label model on both Brier score and log-loss in all three evaluated folds**:

| Fold | Train / test | Brier: meta vs baseline | Log-loss: meta vs baseline | Filter pass | Mean net return: all vs filtered |
|---|---:|---:|---:|---:|---:|
| 1 | 199 / 200 | 0.307512 vs 0.289399 | 0.842559 vs 0.778896 | 26/200 (13.0%) | -0.003107 vs -0.003989 |
| 2 | 399 / 200 | 0.257866 vs 0.247108 | 0.710247 vs 0.687427 | 23/200 (11.5%) | -0.003845 vs -0.005102 |
| 3 | 597 / 200 | 0.259546 vs 0.233835 | 0.713023 vs 0.660552 | 72/200 (36.0%) | -0.003860 vs -0.004282 |

  The filtered mean net return is more negative than the full proposal mean in each evaluated fold. This is a useful negative mechanics result, not a claim about market behavior.
- Governance fields in the artifact: `economic_gates.status=NOT_EVALUATED`, `champion=null`, `promotion_status=BLOCKED_NO_ECONOMIC_GATE`, `trading_authorization=NO_GO`, `risk_gate_approval=false`, and `model_output_is_order_permission=false`.

## Implementation and acceptance checklist

### Method selection and contracts
- [x] MRD Plan B selected in ADR-0005; Phase 5 code does not import DeepAlpha or patch Freqtrade internals.
- [x] Barrier settings, timeframe, fees/slippage, label schema, meta schema and feature names are explicit/versioned.
- [ ] Owner review of ADR-0005 and model-risk scope.
- [ ] Production FreqAI lifecycle integration remains a separately tested future gate; this research service is not registered as a production FreqAI model.

### Triple-Barrier
- [x] Historical-volatility target and first-touch PT/SL or vertical expiry.
- [x] Same-candle PT/SL collision resolves conservatively to stop-loss.
- [x] OHLCV/time checks and shared Phase 3 Data Quality Gate run before Phase 5 feature engineering.
- [x] Outcome returns/costs/barrier label/label-end index are target/outcome columns, not model features.

### OOF and Meta-Label
- [x] Primary predictions are held-out chronological OOF only; primary label horizons are purged before each test fold.
- [x] Meta training checks OOF provenance and uses only mature barrier outcomes before the next test start.
- [x] Meta model persists as versioned JSON, reloads exactly and does not use pickle/joblib.
- [x] Invalid model/input/schema fails closed; filter output contains no entry/exit or risk-approval fields.
- [x] Later-fold OOS score is compared against a historical-rate baseline.
- [x] Exact-head CI and synthetic artifact reviewed for the code SHA stated above.

### Remaining gates
- [ ] Historical OOS evaluation only after the owner supplies authorized data and confirms use rights.
- [ ] Owner-approved economic acceptance criteria pre-registered before any real-data candidate is evaluated.
- [ ] Independent quantitative/model-risk review, realistic transaction-cost sensitivity and production FreqAI lifecycle validation.
- [ ] Owner review and merge of PR #6.

## Safety gates and decision

- All CI is synthetic/mock/offline. No market/provider request, download, exchange connection, order write, notification or host/network operation is part of this Phase 5 run.
- No real, dry-run or paper trading session was started; no order was submitted or cancelled.
- `research_filter_pass` and meta probabilities are diagnostics only, not strategy signals, Risk Gate approval, position sizing authorization or order permission.
- Champion remains null; promotion remains blocked. Economic acceptance and trading authorization remain **NO-GO** regardless of CI outcome.
- Decision: **GO for implementation review only**; **NO-GO for economic acceptance, Champion promotion and trading**.
