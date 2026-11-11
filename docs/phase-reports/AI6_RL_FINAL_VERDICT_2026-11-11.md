# AliX Research Final Verdict — 2026-11-11

## Executive decision

**NO STATISTICAL EDGE FOUND IN RL. Research phase CLOSED.**

The final Owner-authorized AI-6 PPO risk-filter experiment trained for 200,000 timesteps on the immutable pre-hack Nobitex BTC/USDT 1-hour OHLCV dataset. The workflow completed, but PPO accepted zero baseline signal episodes in both validation and final holdout. It did not produce an economically evaluable trading sample.

## Reproducibility

| Item | Value |
|---|---|
| Training commit | `7464e308d4e1efdb573a8225fb3a392ae7614199` |
| Branch | `research/alternative-data-pivot` |
| Standard CI #2220 | [Run 43000569772](https://github.com/angry0tiger-sudo/AliX/actions/runs/43000569772) — SUCCESS |
| AI-6 RL #22 | [Run 43000569350](https://github.com/angry0tiger-sudo/AliX/actions/runs/43000569350) — SUCCESS |
| Dataset | Nobitex BTC/USDT Spot OHLCV, 1h |
| Dataset rows | 31,801 |
| Dataset SHA-256 | `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e` |
| PPO steps | 200,000 |
| Seed | 42 |
| Round-trip cost | 50 bps (25 bps maker fee per side, 0 bps slippage) |
| Policy action space | Discrete(2): SKIP_SIGNAL / TAKE_SIGNAL |
| TAKE reward shaping | +0.25 once per accepted new signal episode; excluded from PnL |

## Final metrics

| Segment | Model | Net PF | Win Rate | Max DD | Statistical Z | Trades |
|---|---|---:|---:|---:|---:|---:|
| Validation | Baseline | 0.3743 | 32.48% | 66.56% | -4.8855 | 234 |
| Validation | PPO | N/A | N/A | 0.00%* | N/A | 0 |
| Untouched holdout | Baseline | 0.4370 | 34.22% | 60.74% | -4.0982 | 225 |
| Untouched holdout | PPO | N/A | N/A | 0.00%* | N/A | 0 |

`N/A` = no completed trades, so PF, Win Rate and Z-Score cannot be calculated.

*PPO remained fully in cash. The 0.00% drawdown is not a successful reduction in drawdown for an operating trading policy.

### Why this is a failure

- Validation: PPO skipped all 234 of 234 candidate signal episodes (0% acceptance).
- Holdout: PPO skipped all 225 of 225 candidate signal episodes (0% acceptance).
- The 0.25 participation bonus was applied zero times in both segments.
- PPO generated zero completed trades, below the required minimum of 51 in each evaluation segment.
- Net PF >= 1.10 and improvement over baseline were not achieved; no PPO PF exists because there were no trades.
- Holdout Z-Score is undefined, not positive.
- The baseline itself is economically weak: PF below 0.5, win rate below 35%, drawdown above 60%, and negative Z-Score in both OOS segments.

The PPO drawdown gate appears to pass only because it never opened a position. This cannot be treated as an improvement over baseline and does not compensate for zero trade activity.

## Promotion checks

| Requirement | Result |
|---|---|
| Validation ≥51 completed PPO trades | FAIL (0) |
| Holdout ≥51 completed PPO trades | FAIL (0) |
| Validation PPO Net PF ≥1.10 | FAIL (N/A) |
| Holdout PPO Net PF ≥1.10 | FAIL (N/A) |
| Validation PF improves over baseline | FAIL |
| Holdout PF improves over baseline | FAIL |
| Positive holdout PPO Z-Score | FAIL (N/A) |
| Validation drawdown below 15% | PASS only because flat |
| Holdout drawdown below 15% | PASS only because flat |
| Paper Trading decision | **NO-GO** |

## Formal closure

**NO STATISTICAL EDGE FOUND IN RL.**

The External Data Pivot remains ABORTED. The research phase is CLOSED on OHLCV-only evidence. No further RL tuning/training or model promotion is authorized. No Nobitex execution adapter preparation or Paper Trading should proceed from this result. A future research attempt requires a new explicit Owner Directive.

## Safety invariants

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `ALIX_DRY_RUN = true`
- `ALIX_ALLOW_LIVE_TRADING = false`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- `nobitex_api_calls_during_rl_run = 0`

No paper or real order was submitted.
