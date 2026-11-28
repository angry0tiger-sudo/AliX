# ADR-076 — M8.1 BTCUSDT One-Hour Public L2 Capture and Correlation

- **Date:** 2026-11-28
- **Status:** ACCEPTED — NO_SIGNIFICANT_OBI_CORRELATION
- **Branch:** `research/nobitex-hf-strategy`
- **Capture commit:** `a0b550f23071f8bf920b866771eb75cba30d8bf6`
- **Capture workflow:** [Run 46106438168](https://github.com/angry0tiger-sudo/AliX/actions/runs/46106438168)
- **Reanalysis workflow:** [Run 46113673381](https://github.com/angry0tiger-sudo/AliX/actions/runs/46113673381)

## 1. Decision

The earlier M1.6 BTCIRT corpus remains ineligible for the BTCUSDT hypothesis under ADR-075. This ADR records a new, independent, public BTCUSDT capture and the pre-existing lightweight M8 statistical analysis executed against its immutable artifact.

**M8.1 result:** `NO_SIGNIFICANT_OBI_CORRELATION`. Neither OBI horizon is significant after the family-wise Bonferroni correction.

This is an exploratory one-hour correlation diagnostic, not evidence of causality, a durable trading edge, execution profitability, or authorization to trade. Non-significance is not evidence that the true relationship is zero. The 15-minute horizon has too few usable samples to estimate correlation.

## 2. Capture provenance and market identity

- Market: **BTCUSDT**
- Source: `NOBITEX_PUBLIC_REST_AND_WEBSOCKET`
- Public REST preflight: `https://apiv2.nobitex.ir/v3/orderbook/BTCUSDT` (no authorization header)
- Public WebSocket channels: `public:orderbook-BTCUSDT, public:trades-BTCUSDT`
- Capture start / finish UTC: `2026-11-28T04:18:02.926+00:00` / `2026-11-28T05:18:08.216+00:00`
- Requested / actual duration: 60 minutes / 60.088 minutes
- Capture status: `NO_GAPS_OBSERVED`
- Order-book / trade / gap rows: 2579 / 93 / 0
- WebSocket order-book / trade messages: 2578 / 73
- Coverage type: `MANUAL_CAPTURE_SEGMENT`
- Capture code SHA: `a0b550f23071f8bf920b866771eb75cba30d8bf6`

## 3. Immutable artifact and integrity

- GitHub raw Artifact ID: `15249319923`
- Raw Artifact name: `alix-m8-1-btcusdt-raw-46106438168`
- GitHub raw Artifact digest: `sha256:bc16983be51490cafebff11c5de06360d6668633bd3a75ea1afbe0defd3837c0`
- Raw retention: 90 days
- Internal analysis ZIP SHA-256: `24459b26500a0216275ada6b60e7f466ab2d179b1b6a3474bb613ae67d984942`
- Manifest SHA-256: `29fe9d3a556d810fc441da9401f529d8efae0c8c507dce88b4a861c69e2a8107`
- Analysis JSON SHA-256: `43dad4753a4ffd23339430b968c5f9f00330a55479f32f5e457f3708f0222363`

The original GitHub Actions artifact was downloaded in a separate run, its manifest and each CSV were checked against the previously recorded SHA-256 values, and the audit script passed before the analyzer ran. The original capture files were not rewritten.

## 4. OBI top-five versus forward mid-price returns

Returns are `10,000 * ln(mid[t+h] / mid[t])` on non-overlapping UTC grids. The OBI feature is top-five displayed depth imbalance. Samples crossing explicit gaps, stale feature snapshots, or late/missing future snapshots are excluded. P-values are two-sided; Bonferroni adjustment is applied across eight feature-horizon combinations.

| Horizon | N | Pearson r | Pearson p | Pearson p (Bonferroni x8) | Spearman rho | Spearman p | Spearman p (Bonferroni x8) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5m | 8 | 0.47108390 | 0.23869631 | 1.00000000 | 0.26190476 | 0.53092286 | 1.00000000 |
| 15m | 2 | N/A | N/A | N/A | N/A | N/A | N/A |

**Interpretation:** neither OBI horizon is significant after family-wise Bonferroni correction. At 5m, N=8 is exploratory and underpowered. At 15m, N=2 and both Pearson and Spearman correlations are not estimable. The result must not be described as zero correlation or a NO-GO conclusion for a tradable strategy.

## 5. Limitations

One hour yields at most about twelve non-overlapping 5-minute and four 15-minute observations before exclusions. Serial dependence and very small N undermine conventional p-value interpretation. Longer, independently preregistered sampling would be required for any confirmatory claim.

## 6. Safety invariants

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- Private API calls: **0**
- Holdout / Post-Hack bytes read: **false / false**
- Paper orders / real order attempts: **0 / 0**
- No model fitting, DeepLOB, or trading execution was performed.
- `main` was not modified.

## 7. Reproducibility

- Capture Run ID: `46106438168`
- Capture Artifact ID: `15249319923`
- Analyzer: `scripts/analyze_m8_microstructure.py`
- Reanalysis Run ID: `46113673381`
- Analysis report SHA-256: `43dad4753a4ffd23339430b968c5f9f00330a55479f32f5e457f3708f0222363`
