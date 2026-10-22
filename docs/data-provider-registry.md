# AliX Data Provider Registry

A public API, public model card or working mock is not approval for commercial processing, historical retention, redistribution or automated trading. No live external provider is activated by Phase 6.

| Provider ID | Kind | Official reference | Use rights / terms | Quota, retention and time semantics | Status |
|---|---|---|---|---|---|
| fixture-rss | Test-only news | Synthetic fixture on .invalid domain | Synthetic fixtures only | Fake transport; no real content | Approved for tests only |
| fixture-market | Test-only market analytics | Synthetic fixture on .invalid domain | Synthetic fixtures only | Fake transport; no real content | Approved for tests only |
| TradingView | Market analytics | https://www.tradingview.com/policies/ | Explicit non-display / algorithmic permission for AliX not evidenced | Plan, quota, retention and timestamp semantics unverified | blocked_for_ai_trading |
| CoinGecko API | Market analytics | https://www.coingecko.com/en/api/pricing | Current tier and use-case terms not reviewed | Provider-specific quota, retention and update semantics required | Conditional |
| CoinMarketCap API | Market analytics | https://coinmarketcap.com/api/documentation/ | Current tier and use-case terms not reviewed | Provider-specific quota, retention and update semantics required | Conditional |
| CoinGlass API | Market analytics | https://docs.coinglass.com/reference/endpoint-overview | Endpoint and plan approval pending | Endpoint-specific quota and timestamps required | Conditional |
| Whale Alert API | On-chain events | https://developer.whale-alert.io/api-account/documentation | Account tier and use-case terms not reviewed | Preserve event time and receive/available time | Conditional |
| X API | Social text | https://help.x.com/en/rules-and-policies/x-api | Data-use and retention approval pending | Contract-specific quota; post and receive times separate | Conditional |
| Telegram API | Social text | https://core.telegram.org/api | Channel rights and use-case approval required | Preserve message time and local receipt time | Conditional |
| Persian news / RSS | News text | Official provider-specific RSS/API URL required before activation | Copyright, processing and retention rights unverified until source is named | Provider-specific polling quota and timestamps required | Conditional |
| News API / economic calendar | News and event data | Provider-specific terms and endpoint required | Commercial-use and retention review pending | Contract quota and event/publication/available times required | Conditional |

## Required evidence before a real source becomes active

Record the provider and official endpoint, legal entity and contract/tier, allowed purpose, commercial and non-display rights where relevant, rate-limit policy, retention/deletion and redistribution terms, timestamp semantics, stable identifiers/revisions, source owner, review date, fallback policy, contract tests and evidence link.

The field license_status=approved means the actual use case has been reviewed and authorized. It must not be inferred from a public endpoint, an API key, a permissive-looking badge, or a successful mock.

Do not scrape, use unofficial endpoints, reuse personal session cookies, rotate accounts/proxies to bypass limits, or ingest private/restricted content without permission. Missing or expired approval fails closed before transport access. External analytics never replace Freqtrade/exchange truth for fills or balances.
