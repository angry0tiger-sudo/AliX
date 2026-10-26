# Owner Directive: Policy-Level Action Logit Masking

Date: 2026-10-26

## Decision

Run #56 showed repeated BUY requests after the position was fully allocated despite the 11-feature observation and the auxiliary cross-entropy loss. Environment-only masking prevented execution but did not constrain the policy distribution. Retraining must therefore use policy-level action masking.

## Required behavior

- Use the observed `current_position_fraction` from the 11-feature observation to mask the raw flattened direction logits before MultiCategoricalDistribution construction, categorical sampling, deterministic selection, PPO log-probability evaluation, and the auxiliary direction-loss path.
- Set BUY logits to `-1e8` when `current_position_fraction >= 1 - 1e-6`, before constructing the categorical distribution.
- Set SELL logits to `-1e8` when `current_position_fraction <= 1e-6`, before constructing the categorical distribution.
- Keep HOLD always legal. Reject missing, malformed, out-of-range, or batch-misaligned mask context; do not silently fall back to an unmasked distribution.
- Preserve the independent environment execution-boundary mask.
- If an invalid BUY nevertheless reaches the environment while BUY is illegal, apply a `-1.0` reward penalty and emit explicit count/total metrics. Existing default penalties for other masked invalid actions remain unchanged.
- Test exact zero probability for forbidden actions in both flat and fully allocated states; test that the fallback penalty is applied without a fill or position change.

## Evaluation and safety acceptance

- Run all existing quality gates without changing their strictness. The research job must wait for CI success on the exact trigger SHA.
- Retrain Recurrent PPO for at least 1,000,000 steps and perform one Validation pass plus one final Holdout pass with Guidance reward and auxiliary optimization disabled. Policy-level Action Masking remains active during both.
- Report deterministic BUY/SELL/HOLD counts, masked requests, fallback penalty counts, fill counts, net profit, drawdown, and win rate from the produced artifact. Do not infer policy learning from reward shaping or environment blocking.
- Preserve `Champion = null`, `order_permission = false`, `trading_authorization = NO-GO`. Zero private API calls and zero real order attempts. No production promotion or real trading is authorized.

## Owner Directive

This policy mask is a safety and action-legality constraint in the research policy, not trading authorization. Any conflicting documentation that claims policy-distribution logits are not masked is superseded by this directive.
