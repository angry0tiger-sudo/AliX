# Phase AI-6 RL — CLOSED: NO STATISTICAL EDGE FOUND IN RL

**Final decision:** CLOSED by Owner Directive dated 2026-11-11, following the completed 200,000-timestep AI-6 PPO risk-filter evaluation.

The 2026-11-09 DEFERRED status was superseded by the 2026-11-11 Owner Directive to resume one final OHLCV-only AI-6 experiment. That final experiment completed successfully as an engineering workflow but failed the economic and activity acceptance criteria. The research workstream is now closed.

## Final run evidence

- Research branch: `research/alternative-data-pivot`
- Training commit: `7464e308d4e1efdb573a8225fb3a392ae7614199`
- AI-6 workflow run #22: [43000569350](https://github.com/angry0tiger-sudo/AliX/actions/runs/43000569350)
- Standard CI run #2220: [43000569772](https://github.com/angry0tiger-sudo/AliX/actions/runs/43000569772)
- Dataset: Nobitex BTC/USDT Spot, 1h, 31,801 pre-hack candles
- Dataset SHA-256: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`
- PPO training: 200,000 timesteps, CPU, seed 42
- Costs: 25 bps maker fee per side, 0 bps slippage, 50 bps round trip

## Final economic result

| Segment / model | Net PF | Win Rate | Max DD | Z-Score | Completed trades | Accepted signal episodes |
|---|---:|---:|---:|---:|---:|---:|
| Validation baseline | 0.3743 | 32.48% | 66.56% | -4.8855 | 234 | 234 / 234 |
| Validation PPO | N/A | N/A | 0.00%* | N/A | 0 | 0 / 234 |
| Untouched holdout baseline | 0.4370 | 34.22% | 60.74% | -4.0982 | 225 | 225 / 225 |
| Untouched holdout PPO | N/A | N/A | 0.00%* | N/A | 0 | 0 / 225 |

`N/A` means there were no completed PPO trades from which to calculate PF, Win Rate or Z-Score.

*PPO's 0.00% drawdown is a consequence of remaining entirely in cash. It is not evidence of lower risk while trading and does not pass the economic promotion gate.

The one-time 0.25 TAKE reward-shaping bonus did not activate: its recorded count and total were both zero in validation and holdout. PPO accepted zero signal episodes despite the full 200,000 training timesteps.

## Promotion decision

**`NO STATISTICAL EDGE FOUND IN RL`**

The run returned `paper_trading_decision = NO-GO`. Both validation and holdout failed the minimum 51-trade support requirement, Net PF >= 1.10, PF improvement over baseline, and positive holdout Z-Score. Drawdown-only checks showed PASS because the policy did not participate; these do not override the failed activity and economic checks.

No further RL tuning, additional training run, model promotion, or Paper Trading preparation is authorized under this research phase. Any future research effort requires a new explicit Owner Directive.

## Closed scope and execution boundary

- External Data Pivot: ABORTED. No Funding Rate, Open Interest, derivatives, sentiment, news, or any external feature is used.
- Research data: Nobitex BTC/USDT Spot OHLCV only, fixed pre-hack dataset.
- OKX is not an execution venue.
- The execution target remains Nobitex Spot, but no execution adapter preparation or Paper Trading is authorized by this failed result.
- No orders were submitted; the AI-6 workflow made zero Nobitex API calls.

## Immutable safety invariants

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `ALIX_DRY_RUN = true`
- `ALIX_ALLOW_LIVE_TRADING = false`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- `nobitex_api_calls_during_rl_run = 0`

The successful CI and training workflow statuses establish implementation and experiment completion only. They do not establish statistical edge or trading authorization.
