# ADR-0014: Nobitex Post-Hack Holdout, Append-Only Integrity and Push-Nonce Dispatch

- Status: Accepted for the active research branch.
- Date: 2026-11-13.
- Active branch: `research/nobitex-hf-strategy`.
- Frozen main: `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb` (unchanged).
- Owner governance source: AliX Master Reference Document v0.9.
- Tracking issue: [#43](https://github.com/angry0tiger-sudo/AliX/issues/43).

## Context and verified violations

Commit `8c17c7766ec3d143eded870f7d2bfd7e0e51d11b` rewrote the historical trial registry instead of preserving it append-only, removing earlier content. The same commit message used `[skip ci]`, preventing normal CI evidence on that exact SHA. Commit `bafcb797c10d4a2b002aa438f6c547509268baf7` also used `[skip ci]` while making a one-line V4-T workflow blob SHA correction.

Commit `5f3fae315c84149f15380c6fb5092915e4af2868` restored the registry to its prior blob `0690f211c719ffb514f760bb59d38a878a9e1358`. The restored historical content is reintroduced only by an explicit append at the end of that baseline. The corrected V4-T workflow blob SHA is `fab1a50d5e6ab174446ab38e8e68afa46516af4e`.

## Decisions: registry and CI integrity

1. `docs/research/TRIAL_REGISTRY.md` is append-only. New records must be appended after existing content. Rewriting, truncating, reordering, or silently replacing historical entries is prohibited.
2. `[skip ci]` is prohibited in all commit messages for this project.
3. CI evidence is accepted only when it belongs to the exact commit SHA under review. A prior green run on a parent SHA is not evidence for a later SHA. Local execution is not CI evidence.
4. Standard CI remains mandatory. Tests, checks, and gates must not be skipped, weakened, xfailed, or relaxed to obtain a green result.

## Dispatch design and frozen-main constraint

GitHub's `workflow_dispatch` registration/UI requires the workflow file to exist on the repository's default branch. The Branch selector chooses the ref to run, but does not remove the default-branch availability requirement. This project keeps `main` frozen, so the dedicated post-hack workflow is activated on the research branch by a narrow `push` trigger filtered to `config/holdout/DISPATCH_TRIGGER`.

The existing HF workflow on this research branch is changed from automatic `push` to owner-only `workflow_dispatch`, preventing the holdout deployment commit from triggering its pre-hack fetch, Binance proxy fetch, or backtest steps. No Binance, Bybit, or HF strategy/fetcher implementation is changed.

A one-file exposure of the dedicated workflow in `main` is a possible future alternative, but is not authorized here. It requires a separate written owner decision; the frozen main SHA remains unchanged.

The dedicated workflow has `contents: read` only, no secrets, and no write-back token. The owner-provided 32-character hex nonce is stored in the V2 commitment and compared with the trigger file. A replay is rejected by checking prior Git history for an earlier nonce value in the trigger file. The nonce is an execution identifier, not an API credential.

## Verified HF workflow evidence classification

- HF #41, Run ID `43274590426`, and HF #45, Run ID `43306747175`, both failed at the identical Step 9: `Fetch Binance Spot proxy 15m candles`.
- The exact same step is the terminal failure in both workflow runs. Classification: **PRE-EXISTING**, outside this holdout-infrastructure scope. It is not repaired or weakened by this ADR.
- Step 8, `Fetch exact pre-hack hourly Nobitex context for TTM-R2`, succeeded in both #41 and #45. This establishes that the public pre-hack Nobitex fetch path was reachable from the Actions runner for those runs; it does not establish reachability of the post-hack window.

## Fixed window decision

The owner-selected start is `2025-07-01T00:00:00Z` / Unix timestamp `1751328000`. The earlier suggested `2025-06-20` start is superseded by the explicit owner value. The selected start is 1,135,800 seconds (13 days, 3.5 hours) after the pre-hack final candle-open timestamp `1750192200`, providing the specified post-incident buffer and preventing timestamp overlap.

The end is fixed at `2026-11-01T00:00:00Z`, not dynamically computed from `now`. Both endpoints are inclusive candle-open timestamps on the UTC hourly grid, producing exactly 11,713 expected candles.

## Acquisition integrity

- Use only the Nobitex public UDF history endpoint `/market/udf/history` already used by the pre-hack fetcher.
- Keep `scripts/fetch_nobitex_public_ohlcv.py` unchanged; use an additive wrapper.
- Pagination maximum is 400 candles per request; UDF `to` is exclusive, so each page uses `page_end + 1`.
- Sort by Unix timestamp and deduplicate identical timestamps before exact-grid validation.
- Zero gap tolerance. Any missing timestamps fail closed and are emitted in a structural gap artifact with the complete timestamp list. The window is never shifted automatically.
- The dataset CSV, if ever acquired, is an Actions artifact. Logs and manifests contain structural metadata only, never price values or derived statistics.
- Artifact retention is set to 30 days; the owner must preserve the artifact and its ID before expiry.
- No training, fitting, tuning, backtest, model evaluation, or holdout consumption is authorized by deployment or PROBE.

## Consequences

- `HOLDOUT_COMMITMENT_V2.md` starts in `PENDING_DISPATCH`; dataset SHA, row count, and acquisition run ID remain pending until a successful owner-triggered acquisition.
- The initial trigger is `PROBE`, which makes exactly one small GET to the public UDF endpoint. The response body is not read, printed, or stored. Only status, endpoint, latency, and run ID are recorded.
- If PROBE returns HTTP 403 or 451, the workflow fails as `GEOBLOCKED`; no proxy, VPN, alternate geography, or bypass is permitted.
- If PROBE returns HTTP 200, that is evidence of endpoint reachability for the small probe request only, not proof of complete historical coverage.
- Acquisition requires the trigger file to contain the committed nonce, V2 status to remain `PENDING_DISPATCH`, and no prior use of that nonce in trigger-file history.
- After acquisition, any transition to verification/final status must be appended to the V2 execution ledger with exact run and artifact provenance.

## Safety invariants

- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`
- `private_api_calls=0`
- `real_order_attempts=0`
- No secrets, no private API calls, no paper orders, no real orders, and no model promotion.
- `main` remains frozen and untouched.


## Addendum — PROBE evidence (2026-11-13)

- Dedicated workflow run #1: Run ID `43360654817`.
- Conclusion: `success`.
- Probe HTTP status: `200`.
- Endpoint: `https://apiv2.nobitex.ir/market/udf/history`.
- Measured request latency: `1290.7 ms`.
- Structural artifact ID: `14007598627`.
- Tested commit: `d2b2672955118291491afd64a731abc70522bd70`.
- The acquisition step was skipped because the trigger remained `PROBE`; no post-hack OHLCV data was fetched.
- This removes the observed runner geo-block concern for the public endpoint probe, but does not prove the full fixed window has complete data.
