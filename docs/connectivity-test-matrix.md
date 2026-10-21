# Connectivity Test Matrix

All Phase 0 checks are mock-only. No live exchange or notification requests are made by CI.

| Case | Expected result | Test method | Status |
|---|---|---|---|
| Route not allowlisted | Deny, no probe | Unit test | Implemented |
| Probe exception | Deny / unhealthy | Unit test | Implemented |
| Approved route reports unhealthy | Deny | Unit test | Implemented |
| Exchange market metadata unavailable | Fail closed | Required future contract test | Pending |
| Proxy/VPN/tunnel secret missing | No connection attempt | Required future test | Pending |
| Bale recipient not allowlisted | Do not send | Unit test | Implemented |
| External service timeout/restart | Record failure, no duplicate delivery | Integration mock required | Pending |