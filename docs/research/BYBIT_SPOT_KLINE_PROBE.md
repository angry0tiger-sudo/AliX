# Bybit Spot Kline Probe — DATA-BLOCKED

- Decision: **DATA-BLOCKED from GitHub Actions runner**
- Probe workflow: `.github/workflows/bybit-spot-kline-probe.yml`
- Workflow blob SHA: `0cf91410653ca0b64213b9df6befe20168d47241`
- Probe commit: `fb9bacf88bab0ae0ee74f7b134e00164e44da0d7`
- Workflow run: [#1 / Run ID 43284092706](https://github.com/angry0tiger-sudo/AliX/actions/runs/43284092706)
- Job ID: `131412768886`
- Probe report schema: `alix.bybit.spot-kline-probe.v1`
- Authentication: none; public GET only.
- Requests: 2; HTTP timeout: 12 seconds each; job timeout: 3 minutes.
- `private_api_calls=0`; `real_order_attempts=0`.

## Request A — recent Spot Kline, limit 5

Exact query:

```text
GET https://api.bybit.com/v5/market/kline
category=spot
symbol=BTCUSDT
interval=15
limit=5
```

Observed response:

- HTTP status: **403**
- Records returned: **0**
- First timestamp: unavailable
- Last timestamp: unavailable
- API retCode: unavailable (HTTP was rejected before a JSON Bybit API result)
- State: `HTTP_BLOCKED_OR_UNAVAILABLE`

Response excerpt returned by CloudFront:

```text
error:The Amazon CloudFront distribution is configured to block access from your country
```

## Request B — exact historical gap window

Exact query:

```text
GET https://api.bybit.com/v5/market/kline
category=spot
symbol=BTCUSDT
interval=15
start=1679661000000
end=1679666400000
limit=1000
```

The requested UTC bounds are:

- Start: `2023-03-24T12:30:00Z`
- End: `2023-03-24T14:00:00Z`
- Expected 15m bar timestamps (inclusive): 7

Observed response:

- HTTP status: **403**
- Records returned: **0**
- Earliest timestamp: unavailable
- Latest timestamp: unavailable
- Exact grid complete: **false**
- State: `HTTP_BLOCKED_OR_UNAVAILABLE`

The probe could not establish whether any of the seven expected bars exist because the request was denied before a candle payload was returned. The report records all seven expected timestamps as unobserved/missing from this response; this does **not** mean the exchange has proven those timestamps absent.

## Conclusion

`DATA_BLOCKED` — both public Spot Kline requests were denied by the CloudFront distribution from the GitHub Actions runner.

This is evidence of **runner-path / geographic access blocking**, not proof that Bybit Spot Kline is globally unavailable. No attempt was made to route around the restriction, use a proxy/tunnel/VPN, change runner geography, or substitute another provider.

Per the owner directive:

1. The HF proxy path is recorded as **DATA-BLOCKED** for this runner environment.
2. Do **not** build a complete Bybit fetcher from this result.
3. Do not splice Binance and Bybit datasets, mask gaps, interpolate, backfill, or shift the historical window.
4. Reconsider the data source only after a separately authorized, policy-compliant access path is evidenced. Any new provider/window requires its own preregistration and integrity checks.

The GitHub Actions workflow conclusion is `success` only because the workflow is designed to preserve a diagnostic report even when the network probe is blocked. It is **not** a successful Bybit connectivity result.

## Safety

- Public GET requests only.
- No API key or authentication header.
- `private_api_calls=0`.
- `real_order_attempts=0`.
- `Champion=null`.
- `order_permission=false`.
- `trading_authorization=NO-GO`.
- `ALIX_DRY_RUN=true`.
- `ALIX_ALLOW_LIVE_TRADING=false`.
