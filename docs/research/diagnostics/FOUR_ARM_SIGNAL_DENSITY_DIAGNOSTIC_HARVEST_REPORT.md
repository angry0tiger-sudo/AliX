# Four-Arm Signal-Density Diagnostic — Harvested Evidence

- Classification: `DIAGNOSTIC_ONLY_NOT_A_TRIAL`
- Harvested at branch HEAD: `7bf32a174e42ea8dee66726593229ed8e79af654`
- Dataset: 27,825-row confirmatory prefix; source SHA-256 `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`
- Economic metrics: **not computed**; trial records: **0**; inference calls in diagnostics: **0**; holdout access: **false**

## Artifact provenance and determinism

| Artifact | Workflow Run | Producer HEAD | Size | API/ZIP SHA-256 | Expiry |
|---:|---:|---|---:|---|---|
| 14171940837 | 43739166184 | `89ce923338ba…` | 4,843 | `f12ce09667fa2768114f99e0fe21b95c4f5b792f16fce9393f3838635dfdf86a` | 2027-02-13T17:05:55Z |
| 14172285941 | 43739941516 | `719b96dfffac…` | 4,843 | `4cfbd156c2d9f02f4e87448b682459b4c521dcab7a4dac1033933824a5b1decd` | 2027-02-13T17:14:04Z |
| 14171877977 | 43740274764 | `0ca123db055b…` | 4,843 | `b094f2f4643a9eed8d3e4770d241e17a39df10fe8cd9e53f4db3f5a6b26f8d28` | 2027-02-13T17:17:35Z |

Each archive contains exactly two files:

| Member | Bytes | SHA-256 |
|---|---:|---|
| `diagnostic-report.json` | 47,323 | `2d759005f54e4008bf504e046f4ba9b2cad397f2092de9887b12110751f4ce2c` |
| `diagnostic-report.md` | 3,404 | `8e89419737f8f6fea6222bdae09930f07a66cacd623c7c9ace77db3c6c81841e` |

The two members are byte-identical across all three artifacts. The ZIP-level SHA-256 values differ, so the three archive containers are not byte-identical; the determinism claim applies to the extracted report payloads, not their ZIP packaging.

## Per-arm × threshold × fold

| Arm | Threshold | Fold | Forecast anchors | Covered bars | Eligible bars | Below threshold | Episodes | ATR rejects | No next bar | Position-open rejects | Completed trades | Risk rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B | 120 | 1 | 121 | 2874 | 0 | 2874 | 0 | 0 | 0 | 0 | 0 | MISSING |
| B | 120 | 2 | 147 | 3500 | 144 | 3356 | 6 | 0 | 0 | 124 | 20 | MISSING |
| B | 120 | 3 | 173 | 4126 | 72 | 4054 | 3 | 0 | 0 | 64 | 8 | MISSING |
| B | 180 | 1 | 121 | 2874 | 0 | 2874 | 0 | 0 | 0 | 0 | 0 | MISSING |
| B | 180 | 2 | 147 | 3500 | 48 | 3452 | 2 | 0 | 0 | 42 | 6 | MISSING |
| B | 180 | 3 | 173 | 4126 | 48 | 4078 | 2 | 0 | 0 | 43 | 5 | MISSING |
| B | 240 | 1 | 121 | 2874 | 0 | 2874 | 0 | 0 | 0 | 0 | 0 | MISSING |
| B | 240 | 2 | 147 | 3500 | 0 | 3500 | 0 | 0 | 0 | 0 | 0 | MISSING |
| B | 240 | 3 | 173 | 4126 | 24 | 4102 | 1 | 0 | 0 | 23 | 1 | MISSING |
| C | 120 | 1 | 121 | 2874 | 96 | 2778 | 4 | 0 | 0 | 82 | 14 | MISSING |
| C | 120 | 2 | 147 | 3500 | 240 | 3260 | 9 | 0 | 0 | 206 | 34 | MISSING |
| C | 120 | 3 | 173 | 4126 | 216 | 3910 | 7 | 0 | 0 | 194 | 22 | MISSING |
| C | 180 | 1 | 121 | 2874 | 48 | 2826 | 2 | 0 | 0 | 43 | 5 | MISSING |
| C | 180 | 2 | 147 | 3500 | 144 | 3356 | 5 | 0 | 0 | 123 | 21 | MISSING |
| C | 180 | 3 | 173 | 4126 | 96 | 4030 | 3 | 0 | 0 | 87 | 9 | MISSING |
| C | 240 | 1 | 121 | 2874 | 48 | 2826 | 2 | 0 | 0 | 43 | 5 | MISSING |
| C | 240 | 2 | 147 | 3500 | 72 | 3428 | 3 | 0 | 0 | 58 | 14 | MISSING |
| C | 240 | 3 | 173 | 4126 | 48 | 4078 | 2 | 0 | 0 | 43 | 5 | MISSING |
| D | 120 | 1 | 121 | 2874 | 48 | 2826 | 2 | 0 | 0 | 43 | 5 | MISSING |
| D | 120 | 2 | 147 | 3500 | 168 | 3332 | 7 | 0 | 0 | 138 | 30 | MISSING |
| D | 120 | 3 | 173 | 4126 | 72 | 4054 | 3 | 0 | 0 | 66 | 6 | MISSING |
| D | 180 | 1 | 121 | 2874 | 0 | 2874 | 0 | 0 | 0 | 0 | 0 | MISSING |
| D | 180 | 2 | 147 | 3500 | 24 | 3476 | 1 | 0 | 0 | 17 | 7 | MISSING |
| D | 180 | 3 | 173 | 4126 | 0 | 4126 | 0 | 0 | 0 | 0 | 0 | MISSING |
| D | 240 | 1 | 121 | 2874 | 0 | 2874 | 0 | 0 | 0 | 0 | 0 | MISSING |
| D | 240 | 2 | 147 | 3500 | 0 | 3500 | 0 | 0 | 0 | 0 | 0 | MISSING |
| D | 240 | 3 | 173 | 4126 | 0 | 4126 | 0 | 0 | 0 | 0 | 0 | MISSING |

## Aggregate forecast-return margin histogram

Relative margin = `(forecast_return_bps - threshold_bps) / threshold_bps`; units are dimensionless. Each row sums to 10,500 covered bars.

| Arm | Threshold | ≤ −100% | (−100,−50] | (−50,−25] | (−25,0] | (0,25] | (25,50] | (50,100] | >100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 120 | 5938 | 3412 | 578 | 356 | 96 | 24 | 72 | 24 |
| B | 180 | 5938 | 3990 | 428 | 48 | 72 | 0 | 24 | 0 |
| B | 240 | 5938 | 4346 | 120 | 72 | 0 | 24 | 0 | 0 |
| C | 120 | 3816 | 5190 | 582 | 360 | 168 | 96 | 120 | 168 |
| C | 180 | 3816 | 5772 | 480 | 144 | 120 | 72 | 72 | 24 |
| C | 240 | 3816 | 6132 | 264 | 120 | 96 | 48 | 24 | 0 |
| D | 120 | 5020 | 4596 | 524 | 72 | 168 | 96 | 24 | 0 |
| D | 180 | 5020 | 5120 | 192 | 144 | 24 | 0 | 0 | 0 |
| D | 240 | 5020 | 5192 | 264 | 24 | 0 | 0 | 0 | 0 |

## Aggregate cell summary and 25%-lower cutoff sensitivity

The 75%-cutoff column is a histogram-derived count of bars whose forecast return would clear a threshold equal to 75% of the listed threshold. It is **not** a rerun: trade counts, position occupancy and PF at those lower cutoffs are uncomputed.

| Arm | Threshold now | Eligible now | Eligible at 75% cutoff | Extra eligible bars | Completed trades (Run #10) | Position-open rejects | Suppressed / eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 120 | 216 | 572 | 356 | 28 | 188 | 87.0% |
| B | 180 | 96 | 144 | 48 | 11 | 85 | 88.5% |
| B | 240 | 24 | 96 | 72 | 1 | 23 | 95.8% |
| C | 120 | 552 | 912 | 360 | 70 | 482 | 87.3% |
| C | 180 | 288 | 432 | 144 | 35 | 253 | 87.8% |
| C | 240 | 168 | 288 | 120 | 24 | 144 | 85.7% |
| D | 120 | 288 | 360 | 72 | 41 | 247 | 85.8% |
| D | 180 | 24 | 168 | 144 | 7 | 17 | 70.8% |
| D | 240 | 0 | 24 | 24 | 0 | 0 | MISSING |

## Bottleneck diagnosis

1. **Signal generation / threshold eligibility is the first bottleneck.** Each cell has 10,500 forecast-covered bars. Only 0–552 bars per cell (0.00%–5.26%) qualify at its registered threshold; 9,948–10,500 bars remain below threshold.
2. **Position occupancy is the second bottleneck.** Of eligible bars, 70.8%–95.8% are suppressed while an existing position is open. Across the nine diagnostic cells, bookkeeping sums to 1,656 eligible bars, 1,439 position-open suppressions and 217 completed trades. This cross-cell sum is descriptive accounting, not 9 independent observations.
3. **Other entry rejects were not observed.** Invalid ATR rejects = 0 and no-valid-next-bar rejects = 0 in all 27 fold rows.
4. **Risk-gate rejects are MISSING, not zero.** The Run #10 runner has no independent pre-entry risk-rejection predicate. ATR stop/target/trailing rules execute after entry; their trade exit outcomes are not risk-gate rejections.

## Interpretation for threshold policy

The distance histogram quantifies the signal supply available if the threshold is lowered. A 25%-relative reduction exposes the candidate bars in the sensitivity table (new cutoffs = 90/135/180 bps for current 120/180/240 bps respectively). This does not translate one-for-one into trades because existing positions suppress later signals and the signal stream is forward-filled across 24 bars.

Run #10's already-observed economics remain adverse to threshold reduction: B NetPF@50 fell from 0.6324 at 180bps / 11 trades to 0.4765 at 120bps / 28 trades; C fell from 0.5240 / 35 trades to 0.4019 / 70 trades. D was nearly flat (0.3973 / 7 versus 0.3981 / 41). All numeric PFs remained below 1.0. No new PF was computed by this audit.

## Cell evaluability and statistical boundaries

The density audit creates no PF values. Run #10 PF evaluability remains manifest-defined at 7/9: B=2/3, C=3/3, D=2/3. B/240 (one trade) and D/240 (zero trades) have PF=null at both costs and remain NOT EVALUABLE. The density data clarifies signal supply and entry attrition but does not create another evaluable PF cell. GLOBAL_N=UNKNOWN; numeric DSR remains MISSING.

## Safety and evidence boundaries

- Reused Run #10 cache: artifact `14163527266`, run `43712086361`, 1,138 anchors, cache SHA-256 `aea6c537f3ca79a28e75faf1a118bd57d5dea4fb58b45e477eb401df03d01868`; no inference was repeated.
- No economic metric, trial ID, new threshold execution, model-weight download, private API call, real-order attempt, or holdout access.
- The diagnostic script/workflow are absent from the harvested HEAD; only the specification and this hashed evidence record remain in-repository.
- No Run #10 manifest, trial-registry result, ADR-042 verdict, or safety authorization was changed.

## Provenance references

- Run #10 source artifact: [14163527266](https://github.com/angry0tiger-sudo/AliX/actions/runs/43712086361).
- Diagnostic run [43739166184](https://github.com/angry0tiger-sudo/AliX/actions/runs/43739166184), artifact `14171940837`, ZIP SHA-256 `f12ce09667fa2768114f99e0fe21b95c4f5b792f16fce9393f3838635dfdf86a`.
- Diagnostic run [43739941516](https://github.com/angry0tiger-sudo/AliX/actions/runs/43739941516), artifact `14172285941`, ZIP SHA-256 `4cfbd156c2d9f02f4e87448b682459b4c521dcab7a4dac1033933824a5b1decd`.
- Diagnostic run [43740274764](https://github.com/angry0tiger-sudo/AliX/actions/runs/43740274764), artifact `14171877977`, ZIP SHA-256 `b094f2f4643a9eed8d3e4770d241e17a39df10fe8cd9e53f4db3f5a6b26f8d28`.

