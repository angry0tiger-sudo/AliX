# Owner Decision Memo — Nobitex 1h Directional Family

- **Date:** 2026-11-15 UTC
- **Decision owner:** Project owner
- **Status:** Options and engineering recommendation only; no owner decision recorded
- **Evidence anchor:** Four-Arm Run #10 / Run ID `43712086361`; result artifact `14163527266`; ADR-042 and ADR-043

## Executive finding

Run #10 produced 0 PASS, 3 FAIL and 9 INSUFFICIENT_SAMPLE. The three failures
are repeated references to one threshold-invariant Donchian control, not
three independent forecast strategies. All nine B/C/D forecast cells have
fewer than 100 completed trades. Every numeric aggregate PF reported in the
cohort is below 1.0; two PF pairs are MISSING because the corresponding cells
have only one trade or zero trades. All 36 fold-stability checks failed.

This supports stopping casual threshold iteration. It does **not** establish
that all nine forecast-driven cells are economic failures, and the newly
adopted Futility predicate in ADR-043 is prospective only.

## Run #10 threshold trend

NetPF values below are aggregate NetPF@50bps.

| Arm | 240bps trades / PF | 180bps trades / PF | 120bps trades / PF |
|---|---:|---:|---:|
| B | 1 / MISSING | 11 / 0.6324 | 28 / 0.4765 |
| C | 24 / 0.5724 | 35 / 0.5240 | 70 / 0.4019 |
| D | 0 / MISSING | 7 / 0.3973 | 41 / 0.3981 |

Best observed NetPF@50bps = **0.6324**; best observed NetPF@90bps =
**0.3936**. No numeric aggregate NetPF reached 1.0. Arm B and C show a clear
inverse relationship across their three tested thresholds; Arm D does not
show a strictly inverse trend (0.3973 at 7 trades versus 0.3981 at 41).
The two missing PF values prevent claiming that every cell has a measured PF.

The signal-density artifact does not identify how many valid forecast bars
crossed each threshold or how many were suppressed by entry/position/risk
gates. The exact location of attrition remains **شاهد کافی نیست** pending the
diagnostic specified in
[`FOUR_ARM_SIGNAL_DENSITY_DIAGNOSTIC_SPEC.md`](../diagnostics/FOUR_ARM_SIGNAL_DENSITY_DIAGNOSTIC_SPEC.md).

## Options

### Option 1 — Lower thresholds to force trades >= 100

**Potential benefit:** More completed trades could satisfy the current sample
gate for B/C/D.

**Evidence against:**
- At 120bps, the maximum completed-trade count among B/C/D is only 70.
- B degrades from 0.6324 at 11 trades (180bps) to 0.4765 at 28 trades
  (120bps).
- C degrades from 0.5724 at 24 trades to 0.4019 at 70 trades.
- Every observed aggregate PF remains below 1.0, and the lower-threshold
  observations do not show an improving economic edge.
- D is nearly flat, not evidence of recovery; at both 7 and 41 trades its PF
  remains approximately 0.398.

**Risk:** Lowering the threshold could increase the sample while simply
confirming a negative edge. It would be a new hypothesis and would require a
new preregistration; no threshold change is authorized here.

### Option 2 — Retire the directional research line and pivot to volatility research

**Potential benefit:** Stop spending cycles on repeated directional threshold
variants and redirect research toward the TTM-R2 volatility signal.

The previously reported TTM-R2 comparison is QLIKE **0.27786 versus 0.39348**
(approximately **29.38% lower**). This is a forecast-quality result, not proof
of trading profitability. The source artifact and protocol must be attached
and reverified in any future preregistration.

**Important distinction:** ADR-043's Futility predicate is future-only and
was not preregistered for Run #10. The current data do not retroactively
trigger a formal `FUTILE` verdict: Arm D is not strictly inverse and two
forecast-cell PF values are missing. The owner may nevertheless choose to
retire the directional workstream as a **portfolio/governance decision**.
That decision must not be described as a retrospective statistical result
under ADR-043.

**Next path if selected:** design and preregister a volatility-centric TTM-R2
study with explicit forecast-to-position translation, costs, execution rules,
fold stability and independent economic gates. Do not open holdout.

### Option 3 — Close the project with a documented NO-GO

**Potential benefit:** A clean stop with no more model, threshold or strategy
experiments.

**Evidence:**
- The owner-provided project summary cites eight directional research paths,
  zero profitable cells and zero stable folds. The eight-path inventory was
  not independently reconstructed in this bounded review; its underlying
  trial IDs/artifacts should remain linked in the project ledger before
  presenting it as a fully re-audited count.
- Run #10 has zero PASS cells, three repeated-control FAIL records, nine
  under-sampled forecast cells and 0/36 stable folds.
- No numeric aggregate PF in Run #10 reaches 1.0; two forecast-cell PF pairs
  are MISSING.

**Boundary:** A project-level NO-GO can be an owner resource-allocation
decision. It is not the same as a statistically established family-wide
economic failure, because nine forecast cells did not meet the sample gate
and ADR-043 cannot be applied retrospectively.

## Engineering recommendation

**Recommend Option 2 as a governance decision, not as a formal ADR-043
FUTILE trigger:** retire the current directional threshold-search workstream,
do not lower thresholds, and move the next research design effort to a
TTM-R2-based volatility hypothesis. The reason is the absence of a measured
positive aggregate edge, the inverse PF/count trend in B and C, zero fold
stability, and the reported independent QLIKE advantage. Keep ADR-043 as a
prospective stopping rule if a new directional family is ever preregistered.

The recommendation is not an owner authorization. Option 3 remains a valid
owner choice if project closure is preferred. Option 1 is the least defensible
use of research budget given the observed PF trend.

## Non-negotiable controls

- `GLOBAL_N=UNKNOWN` remains UNKNOWN.
- DSR remains uncomputed; numeric DSR sensitivities remain MISSING.
- No Run #10 manifest or registry record is altered by this memo.
- No threshold is changed and no new trial is registered.
- No local/remote diagnostic, model inference, or economic dispatch is
  authorized by this memo.
- No holdout access, release, trading permission, or authorization change.
- Frozen `main` remains `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`.

---

## Append-only addendum — TTM-R2 provenance reconstruction (2026-11-15)

The prior QLIKE comparison is now traceable to primary project records rather than an unsupported summary:

- Decision/closure documents: docs/ttm_r2_closure.md and docs/ttm-r2-risk-evaluation.md.
- Exact GitHub execution: [TTM-R2 Risk-Only Research #25, attempt 2 / Run ID 43094730462](https://github.com/angry0tiger-sudo/AliX/actions/runs/43094730462), code commit ad75f44800d3024bc50e7654b1762f49298ef0d9.
- Immutable report artifact: 13916547485, name alix-ttm-r2-risk-only, 510,786 bytes, artifact ZIP SHA-256 15718b3b763ffe8d28d2ee3dd049ba00c2aa71897c6d147fa5d2cf31a58110a8.

### Forecast protocol and exact sample

- Dataset: 31,801 native Nobitex BTCUSDT 1h rows, SHA-256 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e, window 2021-10-31 20:30 UTC through 2025-06-17 20:30 UTC.
- Target: hourly log1p((log(close_t / close_(t-1)))^2); 512 closed hourly context observations; 24 hourly forecast steps. The 24 hourly inverse-scaled variance predictions are summed to form the 24h variance forecast.
- Model: ibm-granite/granite-timeseries-ttm-r2, zero-shot frozen weights, immutable revision d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f, weights SHA-256 a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710.
- Chronological split: 70% training-only scaler fit, 15% validation, final 15% holdout. Forecast origins are 24h apart. The 198 final-holdout decisions run from 2024-12-01 02:30 UTC through 2025-06-16 02:30 UTC; their target horizons end from 2024-12-02 02:30 UTC through 2025-06-17 02:30 UTC. The artifact also contains 198 separate validation forecasts.
- Actual 24h realized variance is the sum of the next 24 squared close-to-close log returns. The baseline is trailing-24h realized-variance persistence: sum of the preceding 24 squared log returns through the forecast origin.
- QLIKE is the mean of actual_variance / forecast_variance - log(actual_variance / forecast_variance) - 1, lower being better. Final holdout: 198 forecasts; TTM QLIKE 0.27785936698219454; persistence QLIKE 0.3934795684925382; improvement 29.3840%. Log-variance MAE also improved (0.63270646 vs 0.69222535).
- The QLIKE forecast comparison is cost-free by protocol; 50 bps round-trip costs were applied only to the separate portfolio replay.

### Purge / embargo clarification

The code creates 24h non-overlapping forecast origins and explicitly checks that every 24h target window lies wholly inside its assigned chronological split. The first validation target begins at the training boundary; the first holdout target begins at the validation boundary. This prevents target windows crossing split boundaries, and the 24h stride prevents overlapping target windows. However, the implementation does not reserve a separate additional 24-bar embargo gap beyond those split-contained target-window constraints. Therefore the accurate description is split-contained, non-overlapping targets; no separately encoded embargo buffer, not an unqualified claim that a conventional purge/embargo gap was applied.

### Economic interpretation and prerequisite for any future pivot

QLIKE improvement is evidence of variance-forecast quality, not proof of trading profitability. The same Run #25 artifact shows the offline 50bps portfolio replay still had 480 holdout trades, NetPF 0.3748 under TTM sizing (0.3731 fixed-size), max drawdown 18.26% (21.63% fixed-size), win rate 31.67%, and net return -18.17% (-21.56% fixed-size). Runtime sizing was not enabled.

Before any TTM-R2 volatility pivot can be preregistered, a new hypothesis must define the forecast-to-position translation (risk multiplier and exposure caps), causal entry/exit and stop behavior, fee/slippage/fill assumptions with the registered 50bps baseline and cost sensitivity, non-overlapping chronological validation with explicit boundary policy, fold-stability and sample-frequency gates, and independent economic acceptance criteria. No pivot, trial, threshold, holdout access, or execution is authorized by this provenance reconstruction.

This provenance is sufficient to cite the QLIKE observation, but it does not authorize a strategy pivot. The owner retains all three project options.

---

## Append-only addendum — harvested signal-density evidence and option reassessment (2026-11-15)

### Evidence harvested

Three separate diagnostic runs (43739166184 / artifact 14171940837; 43739941516 / 14172285941; 43740274764 / 14171877977) each produced the same extracted JSON and Markdown bytes. The ZIP wrappers have distinct SHA-256 values; the internal report file hashes are identical. The complete durable source, 27-fold table and nine histograms are now recorded in:
- docs/research/diagnostics/FOUR_ARM_SIGNAL_DENSITY_DIAGNOSTIC_RESULTS.json
- docs/research/diagnostics/FOUR_ARM_SIGNAL_DENSITY_DIAGNOSTIC_HARVEST_REPORT.md

The audit used only the 27,825-row confirmatory prefix (source SHA-256 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e) and the Run #10 forecast cache (1,138 anchors; cache SHA-256 aea6c537f3ca79a28e75faf1a118bd57d5dea4fb58b45e477eb401df03d01868). It made zero inference calls, computed no PF/PnL/equity/drawdown metric, registered zero trials, and did not access holdout.

### Bottleneck

Each B/C/D × threshold cell has 10,500 forecast-covered bars. Only 0–552 bars (0.00%–5.26%) clear the registered threshold. This is the first attrition point. Among qualified bars, 70.8%–95.8% are suppressed while a position is open. ATR-invalid and no-next-bar rejects are zero. There is no separately instrumented pre-entry risk gate, so risk rejects are MISSING, not zero.

The histogram shows a 25%-relative threshold reduction would add forecast-qualified bars: B +356/+48/+72, C +360/+144/+120, D +72/+144/+24 for the current 120/180/240 thresholds. These are bar counts only. They are not new trade counts, do not account for occupied positions, and have no PF attached.

### Reassessment of the three options

**Option 1 — Lower the thresholds.** The density evidence confirms more forecast bars would qualify at lower thresholds, but not that a 100-trade sample or profitable edge would result. Existing Run #10 economics move against lowering for B and C: NetPF@50 declines from 0.6324 at 180bps / 11 trades to 0.4765 at 120bps / 28 trades for B, and from 0.5240 / 35 trades to 0.4019 / 70 trades for C. D is nearly flat (0.3973 / 7 versus 0.3981 / 41). All numeric PF values remain below 1.0. Do not lower thresholds on this evidence. A lower cutoff would be a new hypothesis requiring separate owner approval and preregistration, with economic gates intact.

**Option 2 — Declare FUTILE and pivot to TTM-R2.** A retrospective FUTILE declaration is not supported. Run #10 is outside ADR-043's prospective scope; the prior evaluability definition yields 7/9 numeric PF cells, but D's NetPF@50 descriptive slope is positive rather than strictly negative. The 25%-lower signal counts add no economic proof. Separately, TTM-R2's 198-holdout QLIKE improvement (0.27785937 vs 0.39347957) establishes variance-forecast quality only; its 50bps holdout sizing replay still had NetPF 0.3748, MaxDD 18.26%, 480 trades, and -18.17% net return. Option 2 remains a permissible portfolio/governance choice to stop directional threshold iteration and prioritize a separately designed volatility-risk hypothesis, but it is not an ADR-043 verdict and does not authorize execution or runtime sizing.

**Option 3 — Close the project with a documented NO-GO.** This remains a valid owner-level resource-allocation choice given 0 PASS, 0/36 stable Run #10 folds, repeated-control FAIL records and nine under-sampled forecast cells. It must not be represented as a statistically proven family-wide failure: two PF pairs are null, and the nine forecast-driven cells did not meet the sample gate.

### Engineering recommendation

Recommend Option 2 as a governance priority only, with a hard boundary: retire casual directional threshold iteration, do not lower the current thresholds, retain TTM-R2 as an offline risk-forecasting result only, and require a new owner-approved preregistration translating volatility forecasts into capped positions, costs/slippage/fills, chronological fold stability and independent economic acceptance. Do not declare Run #10 FUTILE, open holdout, change runtime sizing, dispatch a trial, or arm any execution path.

Option 3 remains the owner's alternative if project closure is preferred. The owner retains the final decision. No option is executed by this memo.

### Accounting and governance boundaries

- Run #10's 12 trial registry records and ADR-042 verdict are unchanged; no new trial record was added.
- Run #10 PF evaluability remains B=2/3, C=3/3, D=2/3; B/240 and D/240 remain PF=null / not evaluable.
- GLOBAL_N=UNKNOWN; DSR numeric values remain MISSING.
- The diagnostic executable and workflow are not restored; only report data, provenance and decision records are added.
- Lock NOT_AUTHORIZED; both triggers PROBE; Champion=null; order_permission=false; trading_authorization=NO-GO.


---

## Append-only addendum — historical multiplicity and DSR input sensitivity (2026-11-15)

### Revised accounting scope

The GitHub tree audit now identifies 27 ml-core-v4 workflow files and 39 research/research-adjacent workflows in the reviewed scope after adding the real Four-Arm experiment workflow to the previously inventoried list. This is a workflow inventory, not 39 independent strategies. The family inventory shows explicit candidate surfaces including V4-AP (36 cells), V4-H (48 experiment rows / 96 weighted model candidates / 864 nested parameter candidates), V4-AD (40 candidate rows), V4-AE (54 trailing configs), V4-P/Q (32 candidate rows each plus nested searches), V4-R (64 candidate rows), V4-V (16 candidate rows plus nested search), and multiple replay workflows over inherited V4-H candidates. Exact lineage and overlap still need reconciliation before any of these counts can be consolidated into a global N.

The previously cited “eight directional research paths” is not a complete workflow/search-surface inventory. It should be described as an earlier bounded grouping, not as the complete number of research workflows or candidate alternatives. The 27 V4 workflow files alone exceed that description; the 39-workflow inventory also includes validation, diagnostic, replay and data-gap workflows that do not all create new strategy trials.

### How a larger true N changes Option 2

If the reconciled historical N is materially larger than the current recorded non-HF subtotal of at least 7,912, the expected maximum Sharpe used by DSR rises and the deflation hurdle becomes harder. This makes a TTM-R2 pivot less defensible as a shortcut around the current directional evidence: TTM-R2's QLIKE improvement is a forecast-quality statistic, not a candidate Sharpe distribution, and its prior costed portfolio replay remained economically negative (NetPF 0.3748 at 50 bps, MaxDD 18.26%, 480 trades, return -18.17%). No retained global per-trial Sharpe series or finite global N exists to quantify its DSR hurdle.

Option 2 remains possible only as a separate owner-authorized research hypothesis with a new preregistration, explicit forecast-to-position translation, execution/cost protocol, candidate-level Sharpe capture and its own DSR cohort. It must not inherit a presumed positive DSR from QLIKE, nor bypass the multiplicity burden by relabeling the same directional search.

### How a larger true N changes Option 3

A substantially larger N, coupled with missing historical Sharpe distributions, increases the archival and statistical burden required to justify further model selection. This strengthens Option 3 as a portfolio/resource-allocation choice if the owner does not wish to fund a complete historical reconstruction and a new, independently justified hypothesis.

That does **not** turn Option 3 into a statistically established family-wide economic NO-GO. Run #10 remains 0 PASS / 3 FAIL / 9 INSUFFICIENT_SAMPLE, the three FAIL rows remain one repeated control, and ADR-043 remains prospective only. A project-level closure may be chosen by the owner, but it must not be represented as a computed DSR failure or a retrospective FUTILE verdict.

### Owner boundary

The engineering recommendation is to freeze further candidate selection until the historical trial/Sharpe crosswalk is either completed or the owner explicitly elects to close the workstream. Option 2 and Option 3 remain owner choices; neither is executed or authorized by this appendix. GLOBAL_N=UNKNOWN and DSR NOT_COMPUTABLE remain unchanged.


---

## Final owner decision addendum — provenance, independence and three governance regimes (2026-11-16)

### Findings that govern all three regimes

This decision addendum incorporates ADR-046. The three independent barriers are:

1. **Multiplicity and Sharpe:** historical \`GLOBAL_N=UNKNOWN\` and global historical \`DSR=NOT_COMPUTABLE\`. The non-HF \`N >= 7,912\` registry subtotal is not a complete global N, and most historical families lack a complete retained Sharpe-per-trial distribution.
2. **Previously consumed data:** AI-6 used the pre-hack data for teacher fitting, PPO training, validation and its terminal 12.5% evaluation. DEAP's OOS began at row 22,308 and ran through the last row, including the entire AI-6 terminal slice. The 27,825-row Run #10 confirmatory prefix is from that same historically used dataset; it is not a fresh untouched holdout.
3. **Pretraining exposure:** Kronos pretraining contains Binance Crypto/Swap observations, but its exact symbol inventory is not public; the technical report gives a June 2024 cutoff, earlier than the post-hack holdout start. Chronos-2-Small lists Chronos/GIFT-Eval datasets and synthetic data, but a complete symbol/date inventory and corpus cutoff are not publicly established. Therefore the clean-pretraining status of the current two-model family is not proven.

The new post-hack window is not in retained evidence as having been consumed by prior AliX trials, but it remains \`FETCHED_LOCKED\`, \`dataset_persisted=false\`, and is only represented by a structural commitment. It is not an already accessible/persisted clean dataset. The exact ADR-036 re-fetch and canonical digest protocol is mandatory after a separate Owner Release Authorization.

### Regime A — one new isolated family on the post-hack holdout

**Evidence for:** a separate Nobitex post-hack window was structurally acquired with 11,700 expected rows and a canonical digest; no retained AliX historical trial record shows it was previously used for model evaluation. The fixed Spec v3 boundary is distinct from the pre-hack data.

**Evidence against / unresolved:** raw OHLCV bytes were not persisted; the holdout can only be re-fetched once under the exact ADR-036 protocol. Kronos has known crypto-class pretraining but no public per-symbol ledger; Chronos-2 lacks a complete symbol/date inventory and a comparable corpus cutoff. Thus the data can be new relative to AliX trials while model pretraining independence remains unproven.

**Mandatory prerequisites:**

- Separate Owner Release Authorization and exact ADR-036 re-fetch: same Nobitex source, Spec v3 timestamps/grid, exactly 11,700 rows, no gaps/off-grid/duplicates, canonical SHA match \`435b313086b78ba8d3caa2e97834c6fabebbbd12eeddcb2865d65982279f33ee\`; any mismatch = STOP.
- A pre-execution corpus decision for every selected model. Require a defensible symbol/date inventory, or explicitly exclude a model from any clean-pretraining claim. The existing provenance file alone does not satisfy this prerequisite.
- New immutable preregistration and canonical trial IDs; a declared integer \`N_budget\`; all arms, models, thresholds, features, labels, seeds, exits, costs and selection rules declared before seeing outcomes.
- Per-candidate net Sharpe and required return-distribution inputs retained from the first candidate onward under one identical protocol.
- Predeclared return-series dependence/correlation treatment and conservative DSR sensitivity. DSR is advisory for this new family only; it does not solve historic global DSR.
- No tuning/retry/second candidate search on the same opened holdout. First outcome inspection consumes its one-time status.

**Cycle cost:** **Very high** engineering/governance cost: complete preregistration, corpus resolution or model exclusion, Owner release, exact one-time data re-fetch/integrity verification, candidate-level ledger/Sharpe capture, and final statistical review. Exact monetary cost and calendar duration: **شاهد کافی نیست**.

**Failure risk:** very high and irreversible for the holdout. A re-fetch/hash mismatch stops the process. Any model-provenance gap, exceeded N budget, missing Sharpe, unhandled dependence, or economic gate failure blocks validation. Once outcome data are examined, the window cannot be recycled as an untouched test for a second selection cycle.

**Weight-artifact deadline:** the retained bundle \`14111134121\` expires at **2026-12-14T11:31:29Z**. At the 2026-11-16T07:41:23+03:30 verification snapshot, approximately **28 days, 7 hours, 20 minutes** remained. If Regime A depends on this exact Actions bundle, its preservation/retrievability decision must be made before expiry; any retained/re-fetched file must match the pinned SHA-256 values. Expiry of this artifact is not an execution authorization and does not alter the holdout lock.

### Regime B — archival reconstruction of historical accounting

**Evidence for:** existing run histories, immutable commits, candidate manifests and retained artifacts may allow partial reconciliation without consuming the post-hack holdout. It could improve the audited lower bound and identify recoverable historical Sharpe outputs.

**Evidence against / unresolved:** there is no complete finite upper bound for historical N, many failed/partial attempts have incomplete candidate progress, and a complete Sharpe-per-trial distribution is absent for most families. Some results may never have been persisted. Exact full recovery: **شاهد کافی نیست**.

**Critical limitation — re-running is not historical reconstruction:** an exact replay of the same candidate on the same data/protocol does not automatically become a new independent candidate under ADR-045; it is a new execution/attempt attached to the existing candidate identity. However, such a replay produces a retrospective metric, not a contemporaneous record of the original selection process, and cannot recreate omitted historical alternatives or what information was available when choices were made. Any changed model, parameter, data, label, threshold, decision rule, or new candidate-selection round is a distinct child trial and must increase the relevant N. Therefore, executing new sweeps or variants to fill missing historical Sharpe values is self-defeating: it expands the selection family while failing to recover the original historical Sharpe distribution. This regime must be archival-only; no rerun, refit, inference, or new candidate selection is authorized by this memo.

**Prerequisites:** all retained workflow-run and artifact manifests; candidate/parent/duplicate crosswalk; genome/config IDs; partial-failure progress evidence; available per-candidate returns/Sharpe; common return-frequency/cost protocol mapping; and reviewer sign-off on unresolved remainders. Missing outputs remain \`UNKNOWN\`, never zero or imputed.

**Cycle cost:** **High to very high** archival/reconciliation cost across 39 research/research-adjacent workflows, nested searches, inherited-candidate replays, DEAP genomes and HF attempts. Exact time/money estimate: **شاهد کافی نیست**.

**Failure risk:** high likelihood of an irreducible unknown remainder; no finite upper N may be established. Even a better N may not make global DSR computable if historical Sharpe series are absent. Weight artifact expiry may remove a convenient immutable copy of the models but does not repair historical Sharpe omissions; expiry is **2026-12-14T11:31:29Z**.

### Regime P — close the project with a documented governance NO-GO

**Evidence for:** the historical family cannot currently satisfy the N/Sharpe evidence contract; the pre-hack data is historically consumed; the new holdout is non-persisted, locked and one-time; and pretraining cleanliness for the current two-model family cannot be proven. This regime avoids turning incomplete evidence into an unsupported profitability claim or consuming the only committed untouched-by-trials window without a defensible protocol.

**Evidence against:** this does not establish that every strategy/model is economically unprofitable. A future, genuinely new dataset and a newly governed family could be studied separately if the Owner chooses to reopen the workstream.

**Required closure wording:**

\`PROJECT_STATUS = NO_GO_DOCUMENTED\`

\`VALIDATION_STATUS = NOT_VALIDATABLE_UNDER_CURRENT_GOVERNANCE / INSUFFICIENT_EVIDENCE\`

\`GLOBAL_N = UNKNOWN\`

\`GLOBAL_DSR = NOT_COMPUTABLE\`

The closure must explicitly say **“non-validatable under current governance”**, not “statistically proven economic failure”. It must not retroactively alter ADR-042/043/045, rewrite the Run #10 mixed cohort, label the directional family statistically FUTILE, or assert a family-wide economic loss not actually established.

**Prerequisites:** Owner approval to close; preserve the immutable historical reports/ADRs and current inventory; retain safety invariants and explicit holdout non-consumption. No new trial or data access is needed.

**Cycle cost:** **Low** relative to Regimes A/B: a final archival closeout/owner decision and preservation of evidence. Exact calendar time or money: **شاهد کافی نیست**.

**Failure risk:** low implementation risk; principal downside is foreclosing further research on this branch and allowing the retained model-weight artifact to expire. If closure is selected, the owner may elect not to preserve artifact \`14111134121\`; it expires on **2026-12-14T11:31:29Z**. This does not affect the already pinned public revision identifiers, but no re-download/reacquisition is implied or performed.

### Engineering recommendation and Owner boundary

**Engineering recommendation: Regime P is the safest and most defensible current choice** if the Owner does not intend to fund the full prerequisites for Regime A. Regime B is worth pursuing only as archival, non-executing reconciliation if the Owner specifically values a better historical lower bound; it cannot manufacture missing contemporaneous Sharpe evidence. Regime A is the only route that could create a new family-scoped validation record, but it has the highest cost, unresolved Chronos-2 pretraining uncertainty, an explicit one-time holdout risk and an expiring retained model bundle. It must not begin before the full prerequisite list and a separate Owner release.

This is an engineering recommendation, **not an Owner decision**. No regime is selected, authorized or executed by this addendum. \`GLOBAL_N=UNKNOWN\` and historical global \`DSR=NOT_COMPUTABLE\` remain unchanged.

