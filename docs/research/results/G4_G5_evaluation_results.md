# AliX Regime R — G4/G5 HGBQ-CQR Pre-hack Evaluation Results

- Decision: **NO-GO / INSUFFICIENT_EVIDENCE**
- Evaluation SHA: `b5d69c9728f7a3ffd3989689fc948fcc96db419c`
- Workflow Run: [44072026184](https://github.com/angry0tiger-sudo/AliX/actions/runs/44072026184)
- Artifact: [14322682950](https://github.com/angry0tiger-sudo/AliX/actions/runs/44072026184/artifacts/14322682950)
- Dataset SHA-256: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`
- G3 configuration SHA-256: `92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa`

## Scope

This was an owner-authorized exploratory evaluation on the fixed pre-hack Nobitex BTC/USDT 1h dataset only. The workflow verified the exact CSV SHA-256 and frozen G3 configuration SHA-256 before fitting. The dataset contains 31,801 rows, from 2021-10-31T20:30:00Z to 2025-06-17T20:30:00Z. All 31,801 volume entries are zero.

## Trial accounting

| Item | Observed |
|---|---:|
| Registered trial slots | 30 |
| Unique candidate configurations | 3 |
| Duplicate slots | 27 |
| Fit calls | 90 |
| Completed trades | 0 |
| Candidates passing all gates at both costs | 0 |
| GLOBAL_N | UNKNOWN |
| DSR | NOT_COMPUTABLE_GLOBAL_N_UNKNOWN |

The 30 registered IDs are not 30 independent configurations. Within each horizon, slots 02–10 repeat slot 01 with the same frozen model settings, seed, features, fold plan, and cost policy.

## Fold-level outcomes

Counts below are per fold and are identical in both 50 bps and 90 bps scenarios.

| Horizon | Forecasts | Signals above cost | Abstentions | Trades | Skipped while occupied |
|---|---:|---:|---:|---:|---:|
| 1h | 743 | 0 | 743 | 0 | 0 |
| 4h | 740 | 0 | 740 | 0 | 0 |
| 24h | 720 | 0 | 720 | 0 | 0 |

Across all 30 slots and both cost scenarios, every tested lower bound was at or below cost. The strict `Lower Bound > Cost` rule therefore rejected every entry. NetPF is N/A because there were no trades. MaxDD of 0.00% only reflects the absence of open positions.

## DSR and verdict

GLOBAL_N remains UNKNOWN because the historical trial multiplicity cohort and dependence structure are not reconciled. DSR is therefore NOT_COMPUTABLE and the positive DSR gate fails closed. The result is **NO-GO / INSUFFICIENT_EVIDENCE**.

## Interpretation limits

The artifact does not store the individual predicted quantiles or conformal interval widths. It cannot distinguish negative lower-bound forecasts from positive lower bounds that remained below cost, nor establish that high uncertainty was the reason for abstention. The zero-volume data limitation is confirmed, but the dataset still contains price-derived features; volume alone cannot be identified as the cause.

## Safety

- Holdout bytes read: false
- Private API calls: 0
- Real order attempts: 0
- Paper orders: 0
- Champion: null
- order_permission: false
- trading_authorization: NO-GO
- main remains frozen at `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`

The detailed per-trial and per-fold JSON/CSV ledgers remain in the linked GitHub Actions artifact.
