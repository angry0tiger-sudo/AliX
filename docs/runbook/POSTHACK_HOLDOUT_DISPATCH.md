# Post-Hack Nobitex Holdout Dispatch Runbook

## Purpose and current state

This runbook controls a one-time, owner-authorized acquisition of the fixed Nobitex BTC/USDT Spot 1h window.

- Branch: `research/nobitex-hf-strategy`
- Workflow: `.github/workflows/nobitex-posthack.yml`
- Trigger file: `config/holdout/DISPATCH_TRIGGER`
- Current trigger value: `PROBE`
- Commitment: `HOLDOUT_COMMITMENT_V2.md`
- Commitment status: `PENDING_DISPATCH`
- Tracking: [Issue #43](https://github.com/angry0tiger-sudo/AliX/issues/43)
- Decision: [ADR-0014](../architecture/decisions/ADR-0014-nobitex-posthack-holdout-push-nonce.md)

The first deployment push runs PROBE only. It does not acquire the historical window.

## Fixed window and integrity rules

- Start: `2025-07-01T00:00:00Z` inclusive.
- End: `2026-11-01T00:00:00Z` inclusive.
- Expected rows: `11,713`.
- Endpoint: `https://apiv2.nobitex.ir/market/udf/history`.
- Timezone: UTC; timestamps represent candle-open Unix seconds.
- Pagination: maximum 400 candles; provider `to` boundary is exclusive.
- Sort ascending and deduplicate exact timestamps before coverage validation.
- Gap policy: zero tolerance. No backfill, interpolation, masking, synthetic fill, or automatic window change.
- Costs/gates are preregistered in `config/holdout/posthack_window_spec.json`.

## PROBE behavior

The initial `PROBE` value causes exactly one small public GET. The workflow does not read, print, or save the response body. It records only HTTP status, endpoint, latency, run ID, commit SHA, and spec SHA.

- HTTP 200: endpoint reachable for the probe request. This is not proof of complete window coverage.
- HTTP 403 or 451: `GEOBLOCKED`; stop. No proxy, VPN, alternate runner geography, or bypass is allowed.
- Other status/network errors: stop and inspect the probe metadata artifact.

## Latest PROBE evidence

The initial PROBE completed successfully on 2026-11-13:

- Run ID: `43360654817`.
- Conclusion: `success`.
- HTTP status: `200`.
- Latency: `1290.7 ms`.
- Artifact ID: `14007598627`.
- The body was not read or retained. The acquisition step was skipped.
- This confirms endpoint reachability for the bounded probe only; full-window coverage remains unverified.

## Owner dispatch: one-time nonce push

Do not use the Actions **Run workflow** button for acquisition. The dedicated workflow also declares `workflow_dispatch` for future availability, but on this frozen-main configuration the reliable acquisition mechanism is the branch-scoped trigger-file push.

1. Open the repository on branch `research/nobitex-hf-strategy`.
2. Navigate to `config/holdout/DISPATCH_TRIGGER` and choose Edit.
3. Replace the exact text `PROBE` with the exact `dispatch_nonce` recorded in `HOLDOUT_COMMITMENT_V2.md`.
4. Commit directly to `research/nobitex-hf-strategy`. That commit changes only the dispatch trigger and starts the dedicated workflow.
5. Do not edit the window boundaries, spec, nonce, or commitment status during this action.

Equivalent Git CLI procedure:

```bash
git checkout research/nobitex-hf-strategy
printf '%s\n' '<dispatch_nonce from HOLDOUT_COMMITMENT_V2.md>' > config/holdout/DISPATCH_TRIGGER
git add config/holdout/DISPATCH_TRIGGER
git commit -m "chore(holdout): dispatch owner-authorized post-hack acquisition"
git push origin research/nobitex-hf-strategy
```

The trigger is accepted only if it matches the committed nonce, V2 remains `PENDING_DISPATCH`, and no earlier commit in trigger-file history used that nonce. A stale or replayed nonce fails closed. Do not retry by reusing it.

## Evidence to return to the owner

After the run, return all of the following:

- GitHub Actions run ID and conclusion.
- Artifact ID and artifact URL.
- Workflow commit SHA.
- Manifest fields: start/end UTC, expected row count, actual row count, gap count, dataset SHA-256, endpoint, run ID, and commit SHA.
- If a gap occurs, the full missing timestamp list from the structural gap artifact.

A gap causes failure. The workflow never changes the requested window automatically. Any replacement window requires a new written owner decision, new immutable spec, new hash, and a new commitment before exposure.

## After successful acquisition

On a later owner-authorized turn, append exact evidence to the V2 execution ledger. The status transition is recorded as `PENDING_VERIFICATION` and then `FINAL` only after artifact/hash/row-grid verification. Preserve earlier commitment and evidence entries; never erase or silently rewrite the original record.

The CSV remains a sealed artifact. No model, researcher, fit, tuning job, backtest, or economic evaluation may read it until a separate owner Release Authorization is explicitly recorded.

## Retention and main-branch alternative

- The workflow artifact retention is 30 days. The owner must download/preserve the original artifact, artifact ID, and manifest before expiry.
- A future one-file disclosure of the workflow in `main` could enable default-branch `workflow_dispatch` discovery in GitHub UI. This is not authorized in this runbook and requires a separate explicit written owner directive. `main` remains frozen.

## Safety invariants

- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `private_api_calls=0`
- `real_order_attempts=0`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`
- No credentials, private API calls, paper orders, real orders, or model promotion.
