# External Data Provider Registry

Status: bootstrap template. Provider use remains disabled until terms, license, limits, retention and operational behavior are verified.

| Provider / dataset | Purpose | Access / terms | License review | Rate limit | Retention | Fallback | Status |
|---|---|---|---|---|---|---|---|
| Exchange OHLCV | Market candles | Provider-specific API terms required | Pending | Pending | Pending | None until approved | BLOCKED |
| Novaryn / Nobitex adapter | Exchange adapter | Canonical URL and pinned commit not established | Pending | N/A | N/A | No adapter selected | BLOCKED |
| Hugging Face model | NLP inference | Exact model ID and revision must be selected | Pending per model | Provider-specific | Must be defined | Neutral/abstain | BLOCKED |
| Bale Bot API | Daily report delivery | Official API contract review required | Pending | Pending | Do not retain secrets in logs | Local report artifact | MOCK ONLY |

Do not interpret an empty or pending registry cell as approval. Verify each entry before enabling a provider.