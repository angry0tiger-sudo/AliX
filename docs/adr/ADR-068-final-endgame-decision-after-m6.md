# ADR-068 — AliX Endgame Decision After M6 Closure

- **Status:** M6 closure confirmed; project endgame choice **PENDING OWNER DECISION**
- **Date:** 2026-11-27
- **Repository:** `angry0tiger-sudo/AliX`
- **Branch:** `research/nobitex-hf-strategy`
- **Base HEAD reviewed:** `47e7bc7f70f8a019f3ac61fee2898da7b73dc2eb`
- **Latest Standard CI on base HEAD:** [#2644 / Run 46022780142](https://github.com/angry0tiger-sudo/AliX/actions/runs/46022780142) — SUCCESS
- **Related decisions:** ADR-053–ADR-067; TTM-R2 closure record `docs/ttm_r2_closure.md`
- **Scope:** Record M6 closure, consolidate the four failed strategy families and assess the two endgame options. This ADR does not authorize a new experiment.

## 1. Decision recorded

The Owner has accepted closure of M6. The two preregistered three-class Random Forest candidates are **CLOSED — INSUFFICIENT_SAMPLE / NO-GO**:

- `NOBITEX-M6-1h-RF-3CLASS-01`
- `NOBITEX-M6-4h-RF-3CLASS-01`

The successful evaluation is [M6 workflow Run 46021926723](https://github.com/angry0tiger-sudo/AliX/actions/runs/46021926723), on the locked preregistration SHA-256 `f94af9eb6f6003818beb88fcdbfbebe2ac2d7ac8a42281529db2781b44962e42`. The evaluation artifact is `alix-m6-btcusdt-three-class-results`, Artifact ID `15210185201`.

This is a scientific/economic closure of the registered M6 design, not a claim that every possible three-class target or every trading strategy is impossible. No retry, threshold change, parameter sweep, feature ablation, or replacement estimator is authorized by this ADR.

The broader project endgame remains undecided. The two alternatives are:
1. One final, tightly bounded Donchian Breakout + TTM-R2 volatility-sizing research test.
2. Formal project-level NO-GO for further strategy research and archival closeout.

Neither alternative is selected by this ADR. The current state remains `OWNER_DECISION_PENDING`.

## 2. Consolidated record of the four failed strategy families

The table below records the ten distinct logical candidate/experiment identities that were in the v1.2 inventory before M6, plus the two M6 candidates now closed. These are identity counts, not a certified global statistical trial count. The pre-M6 total is 10 (HGBQ-CQR 3 + OFI 2 + binary Regime T 1 + M5.2 4); M6 adds 2, giving **12 distinct identities across these focused workstreams**. This must not be confused with workflow attempts, fit counts, fold counts, duplicate registry slots, or the full AliX historical search space.

| Family | Logical identities | Dataset / scope | Main observed evidence | Disposition |
|---|---:|---|---|---|
| HGBQ-CQR / Regime P | 3 unique horizon candidates; 30 registry slots including 27 duplicates | Nobitex BTCUSDT PRE-HACK 1h OHLCV; horizons 1h, 4h, 24h | 0/3 calibrated lower bounds cleared either Cost-50 or Cost-90; no hypothetical trade was admitted. The G4.6 volume ablation did not improve the G4.5 result; its 90/90 diagnostics were identical. | CLOSED — NO-GO |
| M3 OFI / Regime M | 2 (1h, 4h) | Five historical capture segments of BTCIRT, not BTCUSDT | 1h: 665 cohorts, NetPF 0.07347, PnL -7.9225%. 4h: 82 cohorts, NetPF 0.23269, PnL -0.1214%; stability 0/4. | CLOSED — NO-GO; not transferable as BTCUSDT evidence |
| Regime T binary Triple-Barrier + Logistic Regression | 1 | BTCUSDT PRE-HACK; 24h vertical barrier; three chronological folds | 31,801 input rows; 24,740 binary events; 7,061 censored/ambiguous/unavailable observations excluded; 1,652 pooled OOF predictions. Accuracy 51.09%, balanced accuracy 51.57%, below the preregistered strict >55% screen. This was a classification screen, not an economic backtest. | CLOSED — NO-GO |
| M5.2 Sentiment | 4 (Control/Sentiment × 1h/4h) | Nobitex BTCUSDT; 16 fold fits; 101,648 OOS prediction rows | All eight NetPF results were below 1; stability 0/4. Sentiment underperformed its paired Control in all four comparisons, including net performance. | CLOSED — NO-GO |
| M6 three-class Triple-Barrier + Random Forest | 2 (1h, 4h) | 31,801-row committed BTCUSDT PRE-HACK 1h file; 4h deterministically aggregated; 8 fits; 31,641 OOS predictions | Both candidates had 2/4 valid folds and 0/4 stable folds. NetPF remained below 1 and MaxDD exceeded 15% at both costs; see the detailed table below. | CLOSED — INSUFFICIENT_SAMPLE / NO-GO |

### 2.1 M6 economic results

| Horizon | Cost | NetPF | Completed OOS trades | MaxDD | Valid folds | Stable folds |
|---|---|---:|---:|---:|---:|---:|
| 1h | 50 bps | 0.160856 | 1,177 | 22.58% | 2/4 | 0/4 |
| 1h | 90 bps | 0.013034 | 1,177 | 41.29% | 2/4 | 0/4 |
| 4h | 50 bps | 0.534557 | 501 | 28.18% | 2/4 | 0/4 |
| 4h | 90 bps | 0.278479 | 501 | 56.06% | 2/4 | 0/4 |

The aggregate trade-count minimum was met, but valid-fold and stable-fold gates failed. A zero-trade fold is not evidence of zero risk or profitability. No M6 horizon/cost combination met the registered economic requirements.

### 2.2 Counting and interpretation boundary

- The **ten** historical candidate/experiment identities refer to the four workstreams listed before M6 in MRD v1.2.
- M6 contributes **two additional logical candidate identities**. The first failed workflow attempt stopped at trigger-scope validation before data access or fit and is an infrastructure attempt, not another candidate.
- The v1.2 registry baseline contained 37 slots, including 27 HGBQ duplicate slots. M6 candidate identities are appended separately; registry rows, candidates and workflow attempts must not be conflated.
- `GLOBAL_N=UNKNOWN` and `GLOBAL_DSR=NOT_COMPUTABLE` remain unchanged. The inventory above is not a complete reconstruction of every historical AliX experiment (including older AI-6/PPO, DEAP, Chronos and other workstreams).
- M3 OFI used BTCIRT. It cannot be used to claim that an OFI strategy was tested and rejected on BTCUSDT.

## 3. TTM-R2 evidence: forecasting value is not entry alpha

TTM-R2 remains the only evaluated component in this endgame discussion with a clear positive forecasting result:

| Final 1h holdout metric | TTM-R2 | 24h persistence baseline |
|---|---:|---:|
| QLIKE | 0.27785937 | 0.39347957 |
| Log-variance MAE | 0.63270646 | 0.69222535 |
| Non-overlapping forecasts | 198 | 198 |

QLIKE improved by 29.38%. The forecast gate was `PASS_RISK_FORECAST_RESEARCH_ONLY`. This result supports the limited claim that the frozen zero-shot TTM-R2 model forecast this variance target better than the registered persistence baseline. It does not establish trade direction, Donchian signal quality, net profitability, or readiness for deployment.

The existing offline TTM position-sizing replay already tested whether better variance forecasts could improve a weak fixed entry teacher. Both portfolios used the same 480 trade events and 50 bps round-trip cost:

| Metric | Fixed 10% size | TTM risk-sized |
|---|---:|---:|
| NetPF | 0.3731 | 0.3748 |
| MaxDD | 21.63% | 18.26% |
| Win rate | 31.67% | 31.67% |
| Trade count | 480 | 480 |
| Net return | -21.56% | -18.17% |

The sizing overlay reduced MaxDD by 3.37 percentage points and improved NetPF by only 0.0017, while leaving win rate and trade count unchanged. The risk-managed portfolio remained materially unprofitable and above the 15% drawdown objective. Thus, TTM sizing has demonstrated a modest risk-shaping effect, not an economic edge capable of rescuing an unprofitable entry process.

Reference: [TTM-R2 phase closure](docs/ttm_r2_closure.md), based on workflow [Run 43094730462](https://github.com/angry0tiger-sudo/AliX/actions/runs/43094730462).

## 4. Option 1 — One bounded Donchian Breakout + TTM volatility-sizing test

### 4.1 Why the hypothesis is distinct

This option would replace the failed ML directional classifiers with a deterministic price-channel entry rule. TTM-R2 would remain a risk-only component and could alter position size only after a Donchian entry independently exists.

The causal responsibilities must remain separate:

- **Donchian:** generates a long entry from a prior-bar channel breakout. TTM must not create, suppress, rank or reverse the entry.
- **TTM-R2:** supplies an as-of 24-hour variance forecast and scales the size of an already-valid Donchian trade.
- **Risk controls:** impose an independent hard cap; the forecast cannot raise exposure above the fixed-size reference.

The key hypothesis is not “TTM makes a losing system profitable.” It is: “A fixed Donchian trend-following rule has a measurable BTCUSDT net edge, and a point-in-time TTM sizing overlay may improve its risk-adjusted economic profile without changing its entries.”

### 4.2 Evidence in favor

1. TTM-R2 passed its registered variance-forecast gate on 198 non-overlapping 24h holdout origins (QLIKE improvement 29.38%).
2. Trend-following entries are structurally different from the supervised direction/event targets that failed in HGBQ-CQR, Logistic Triple-Barrier, and M5.2.
3. The TTM replay showed a small drawdown reduction with identical entries, which is consistent with a risk-allocation use case even though it did not meet economic gates.
4. A paired fixed-size versus TTM-sized replay can isolate the marginal contribution of sizing instead of attributing every change to a different entry or exit rule.

### 4.3 Reasons for caution

1. **The underlying Donchian entry edge is unproven on Nobitex BTCUSDT 1h/4h.** The existence of Donchian code or a successful unit test is not an economic result.
2. The existing `research/ai_ensemble/hf_donchian.py` helper is restricted to native 15m/30m input. It uses Donchian-20, ATR-14, a 2×ATR stop, a 4×ATR target and a 50 bps round-trip cost assumption. It is not a verified 1h/4h BTCUSDT experiment.
3. That helper clips the TTM position multiplier to **0.5×–1.5×**, so it can raise exposure 50% above the fixed-size reference. This conflicts with the earlier TTM-R2 risk-only constraint that the multiplier must never exceed **1.0×**. The helper must not be treated as an approved sizing contract or reused unchanged for this option.
4. The prior TTM sizing replay remained unprofitable: NetPF 0.3748 and MaxDD 18.26%. It does not justify expecting a positive outcome from a new entry rule.
5. The committed PRE-HACK file has been repeatedly exposed to research, including the TTM sizing replay and M6. Any new use of it is exploratory, not a fresh confirmatory holdout. Post-Hack Holdout remains sealed.
6. Volatility targeting is not universally beneficial. Published research finds that its performance depends on the asset, strategy and trend/volatility relationship; forecast accuracy alone does not guarantee improved portfolio outcomes. See [Hood & Raughtigan (2025), “Volatility Targeting Is Trendy”](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4773781) and [Cederburg et al. (2020), “On the Performance of Volatility-Managed Portfolios”](https://www.lehigh.edu/~xuy219/research/COWY.pdf).

### 4.4 Minimum conditions if Option 1 is selected later

This ADR does not authorize or preregister the experiment. A separate owner-approved preregistration would need to freeze, before any run:

- BTCUSDT only; separate 1h and 4h decision cadences, with 4h derived deterministically from the committed 1h bars if that is the selected source contract.
- One fixed Donchian channel length, one entry/exit geometry, one position sizing formula, one risk reference and one set of execution/fill assumptions. No parameter sweep or post-result tuning.
- A point-in-time TTM forecast lookup strictly available by the trade entry timestamp.
- A sizing cap of **0 < multiplier ≤ 1.0×** unless the Owner explicitly approves a different risk budget in a separate decision. No leverage or volatility-driven increase above the fixed-size reference.
- A paired control: identical Donchian entries, exits and cost assumptions under fixed sizing versus TTM sizing. This is necessary to isolate sizing value.
- Separate reporting of NetPF, completed trades, MaxDD, net return, turnover, exposure, trade frequency and fold-by-fold stability under Cost-50 and Cost-90; retain the established project gates (NetPF ≥ 1.10, at least 100 completed costed OOS trades, MaxDD ≤ 15%, and at least 3/4 stable temporal folds). The TTM closure's stricter planning target of >200 trades/year must also be considered before any release review.
- Explicit classification as PRE-HACK exploratory evidence. A positive result cannot be called confirmatory and cannot open Holdout or confer Paper/Live authority.
- A separately reviewed fee/fill contract. Cost-50 and Cost-90 are registered stress scenarios, not proof of realized costs. Commission tier, maker/taker mix, spread, market impact, partial fills, latency and slippage must not be silently assumed away.

**Assessment:** Option 1 has a coherent, materially distinct hypothesis and may be worth exactly one bounded exploratory test if the Owner wants one last strategy experiment. It is not currently supported by economic evidence and should not be framed as the next active model track.

## 5. Option 2 — Formal project-level NO-GO and archive

This option closes further strategy/model research under the current AliX scope, preserves the repository, ADRs, workflows, registry, source data commitments and artifacts, and stops spending the existing PRE-HACK sample on further variants.

Its defensible public conclusion must be carefully scoped:

> No tested and preregistered candidate in the evaluated BTCUSDT workstreams met the project's registered economic acceptance criteria on the available PRE-HACK evidence. The M3 OFI result is BTCIRT-only and is not evidence about BTCUSDT. Therefore, no Champion or deployable edge has been established for the tested scope.

It would be scientifically incorrect to assert that “no economic edge exists on Nobitex 1h/4h with public data and real costs” as a universal fact. The finite set of candidates cannot prove non-existence for every possible strategy. In addition, Cost-50 and Cost-90 are modelled round-trip scenarios; without a complete, verified fee-tier and order-fill record, they must not be described as measured realized trading costs. The correct closure statement is **NO EDGE PROVEN UNDER THE TESTED CANDIDATES AND REGISTERED COST/FILL ASSUMPTIONS**, not a universal impossibility theorem.

**Assessment:** This is the strongest option for scientific discipline and cost control. It avoids another exploratory variant on a heavily reused historical sample and prevents a positive in-sample discovery from being mistaken for independent confirmation.

## 6. Recommendation

**Recommendation: Option 2 — archive the current strategy research program as NO-GO**, unless the Owner places substantial value on one final, explicitly exploratory Donchian experiment.

Reasons:
- TTM-R2 forecasting success is real but narrow; its prior sizing replay did not approach the economic gates.
- The existing Donchian helper is not a 1h/4h BTCUSDT economic result and its 1.5× sizing ceiling conflicts with the TTM risk-only cap.
- The PRE-HACK evidence has been consumed by multiple related experiments; another positive result would still be exploratory.
- No measured execution-cost evidence currently justifies describing Cost-50/90 as actual realized costs.
- The project already has four materially different failed strategy families and no Champion.

Option 1 remains a possible final *research-only* branch if selected explicitly. It must not be opened automatically by this recommendation or by the existence of old helper code.

## 7. Current governance and safety state

Until a later Owner decision is recorded:

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `Holdout = SEALED`
- `posthack_data_read = false`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- `paper_orders = 0`
- `main = FROZEN`; no content or ref update is permitted.
- No code, model, feature, adapter, data acquisition, model fit, inference, backtest, Paper or Live workflow is authorized by ADR-068.

If Option 1 is selected, only drafting a new preregistration is the next allowed step until a separate execution authorization is granted. If Option 2 is selected, the next step is a documentation-only archival closeout and a final repository inventory; it must not delete historical evidence or modify `main`.

## 8. Owner decision required

**Endgame status: `OWNER_DECISION_PENDING`.**

- **Option 1:** one final, bounded Donchian Breakout + TTM-R2 sizing research experiment, with a new preregistration and an exploratory-only PRE-HACK evidence label.
- **Option 2:** formally close the current AliX strategy research program as NO-GO and archive the evidence.

This ADR records the facts, boundaries and recommendation. It does not choose on the Owner's behalf.

Architecture First. Evidence Always. Fail Closed.
