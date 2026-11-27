# ADR-074 — Formal Closure of M7 Cross-Sectional Ridge

- **Date:** 2026-11-27
- **Status:** ACCEPTED — CLOSED_NO_GO
- **Scope:** M7 Cross-Sectional Ridge, revisions G2.0 and G3.0
- **Branch:** `research/nobitex-hf-strategy`
- **Owner decision:** The owner has explicitly accepted the G2.0 and G3.0 NO-GO outcomes and directed the project to proceed to M8.

## 1. Decision

The M7 Cross-Sectional Ridge research path is formally closed as **CLOSED_NO_GO**. The project will not continue feature sweeps, model changes, universe changes, threshold searches, or re-runs of M7 under these trial identities. A future, materially different idea must receive a new preregistration and must not overwrite this negative result.

G2.0 and G3.0 remain separately recorded:
- **G2.0:** feature-quality / empty-panel NO-GO. The diagnostic established that `relative_volume_shock_24h` caused 100% row rejection because every sampled prior-24h median volume was nonpositive. `beta_24h_to_btc` rejected 0% of rows.
- **G3.0:** feature-only supersession replacing the volume-dependent feature with `log_return_24h = ln(close[t] / close[t-24])`. This repaired panel availability, but the locked economic evaluation returned NO-GO.

## 2. G3.0 final economic result

Source: M7 Economic Evaluation Run #4, ID `46060086225`, using the exact implementation/config parent `ba8977e86b2414674fc30e67bdeb6a97b357fa65`. Its exact-parent Standard CI was Run #2689 / ID `46059360239`, conclusion SUCCESS.

| Candidate | Cost | OOS trades | NetPF | MaxDD | Valid folds | Stable folds | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 5PAIR G3.0 | 50 bps | 4,565 | 0.503741 | 99.997453% | 4/4 | 0/4 | FAIL |
| 5PAIR G3.0 | 90 bps | 4,565 | 0.262960 | 99.9999997% | 4/4 | 0/4 | FAIL |
| 9PAIR G3.0 | 50 bps | 5,959 | 0.424442 | 99.999872% | 4/4 | 0/4 | FAIL |
| 9PAIR G3.0 | 90 bps | 5,959 | 0.193591 | 99.9999999992% | 4/4 | 0/4 | FAIL |

Every scenario failed the preregistered NetPF >= 1.10, MaxDD <= 15%, and stable-fold >= 3/4 gates. Four of four folds were valid, but none were stable. The family verdict is **NO-GO**. The complete diagnostic and result evidence remains in [ADR-073](ADR-073-m7-feature-availability-diagnostic.md) and the M7 result artifacts.

## 3. Trial accounting

No prior M7 identity or configuration revision is deleted, relabelled as a pass, or silently merged. G2.0 and G3.0 remain in the historical trial record for later multiple-testing / DSR accounting. This closure does not assert a known global experiment count: `GLOBAL_N=UNKNOWN`; `GLOBAL_DSR=NOT_COMPUTABLE`.

## 4. CI and repository state

- Final M7 branch cleanup Standard CI: [Run #2693 / ID 46060402042](https://github.com/angry0tiger-sudo/AliX/actions/runs/46060402042), SHA `eb0948754fba7167e9a6751c9a3e86f3fc75a930`, conclusion **SUCCESS**.
- G3.0 exact-parent Standard CI: [Run #2689 / ID 46059360239](https://github.com/angry0tiger-sudo/AliX/actions/runs/46059360239), conclusion **SUCCESS**.
- The current M8 implementation work is committed only to `research/nobitex-hf-strategy`.
- Protected `main` remains frozen at `a1c1a658906cd79be62796885e3ed782b66a267d`; tree SHA `5761c6c34143f8e6924f53a4e745c48932a5f791`.

## 5. Safety boundary

No Holdout or Post-Hack bytes were read. No paid service, wallet, TinyFish, private API, paper order, or real order was used. `Champion=null`, `order_permission=false`, and `trading_authorization=NO-GO` remain unchanged.

## 6. Consequence

M7 is closed. The next active research path is M8 L2 microstructure, subject to its own correct-market data-eligibility gate. M1.6 was verified as BTCIRT and cannot be used as BTCUSDT evidence; see [ADR-075](ADR-075-m8-microstructure-data-market-eligibility.md).
