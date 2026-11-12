# QWEN.md — AliX Project Context & Non-Negotiable Rules

## State (FROZEN)

Paper-only research bot. Venue: Nobitex Spot (long-only, no leverage/short).

Freqtrade 2026.7 pinned, Python 3.12.

Champion=null; order_permission=false; trading_authorization=NO-GO;
ALIX_DRY_RUN=true; ALIX_ALLOW_LIVE_TRADING=false; private_api_calls=0; real_order_attempts=0.

`main` is FROZEN by standing owner directive; develop on the active research branch.

## Prime directive

No "guaranteed profit" claims. Goal: discover a validated economic edge IF one exists,
else protect capital. A documented NO-GO is a valid outcome.

## Safety invariants (never relax/bypass)

1. Fail-closed: missing/invalid data/model/schema/permission => NO trade.
2. Risk Plane independent of models; absolute caps + latched kill switch override confidence=100%.
3. Never enable live / set order_permission=true / promote Champion. Requires owner Release Authorization.
4. Secrets never in code/chat/logs/commits/artifacts. Public data needs no key.
5. No real orders / private API calls in research or CI.

## Data rules

- Nobitex-native for FINAL validation. Binance/Bybit are research proxies only; no cross-venue
  transfer assumption without a transferability test.
- Every dataset: manifest + SHA-256 + exact timestamp grid + continuity; NO backfill/interpolation/
  forward-fill/synthetic gap repair; never mask gaps.
- Bybit V5 Kline proxy policy: Spot only; 15m/30m intervals; explicit start/end; at most 1,000
  records per request; responses are reverse-ordered and MUST be sorted before validation;
  require the exact complete grid and zero gaps; emit manifest + SHA-256; never mask gaps.
- Point-in-time: features past-only; targets separate; purge+embargo; scaler train-only; no look-ahead.
- Do not assume 30bps live fee; use owner-approved cost; report 50 & 90 bps (maker/taker);
  Nobitex has no post-only guarantee.

## Evidence protocol

Pre-register hypotheses; log EVERY trial (DSR/PBO); final holdout committed by hash, opened once;
absolute gates NetPF>=1.10, trades>=100, MaxDD<=15%, DSR>0, stable >=3 folds; paper promotion via
anytime-valid confidence sequences.

## Engineering gates (never weaken)

ruff, mypy strict, pytest cov>=90%, import-linter, xenon, bandit, pip-audit. PR-only; human review.

## Must NOT

Invent/label "guaranteed profit"; relax gates/disable tests/bypass fail-closed; touch secrets;
enable live; auto-promote; silently overwrite docs (record ADRs); touch frozen main.
