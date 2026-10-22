# هدف و قواعد اجرای نقشهٔ راه
این سند ترتیب اجرای پروژه را مشخص می‌کند. «پایان فاز» به معنی گذشتن از معیار پذیرش همان فاز است، نه صرفاً پیاده‌سازی فایل‌ها. هیچ فاز شکست‌خورده‌ای با تیک زدن ظاهری بسته نمی‌شود. سرمایهٔ واقعی تا فاز ۱۱ ممنوع است.

## اصول کنترل اجرای پروژه
- هر تغییر با PR، بازبینی، آزمون‌های اتوماتیک و ثبت ADR مرتبط وارد شاخهٔ اصلی می‌شود.
- نسخهٔ Freqtrade، Python، CCXT، dependencyها، model revision و دادهٔ ورودی در هر آزمایش ثبت و pin می‌شوند.
- توسعهٔ موازی مجاز است؛ اما release gateها ترتیب وابستگی‌ها را تغییر نمی‌دهند.
- نتیجهٔ هر فاز باید شامل artifact قابل بازتولید، گزارش تست، risk review و تصمیم `GO / NO-GO` باشد.
- شکست معیارهای امنیت، داده، قرارداد صرافی یا بازیابی، توقف فاز و منع promotion به فاز بعد است.
- مالک مرجع نهایی تصمیم‌های داخلی است؛ دستور متعارض با سند به Owner Directive/PR نسخه‌بندی‌شده تبدیل می‌شود. فعال‌سازی live فقط با Release Authorization و عبور گیت‌های اجباری ممکن است.
- در فاز صفر `reporting.timezone` (پیش‌فرض `Asia/Tehran` و قابل‌تغییر) و chat allowlist بله تعیین می‌شود؛ token فقط از secret manager/local secret mount خوانده می‌شود.

# گردش کار End-to-End در GitHub و انتقال به سیستم محلی

این پروژه با GitHub به‌عنوان محل مدیریت سورس، مستندات، PRها و شواهد CI توسعه می‌یابد؛ **کلید واقعی صرافی و اجرای زنده در GitHub-hosted runners انجام نمی‌شود**. GitHub محل کنترل تغییرات است، نه محل نگهداری سرمایه یا secretهای معامله.

## ساختار اجرایی پیشنهادی
- `main`: فقط نسخه‌های بازبینی‌شده و قابل بازتولید؛ branch protection، حداقل یک approval برای تغییرات ریسک/اتصال/صرافی، و CI سبز الزامی است.
- `feat/<scope>`, `fix/<scope>`, `docs/<scope>`: تغییرات کوچک؛ هر تغییر مرتبط با یک GitHub Issue/Phase ID و ADR است.
- هر فاز Roadmap یک Epic/Issue دارد و معیار پذیرش آن به taskهای کوچک قابل تست تبدیل می‌شود. یک فاز فقط با گزارش `GO / NO-GO / CONDITIONAL GO` بسته می‌شود.
- PR باید شامل هدف، تغییر معماری، tests, data/license impact, risk impact, rollout/rollback و evidence links باشد. هر تغییر در provider، transport، model یا risk policy، تست‌های contract/security/recovery مربوطه را دوباره اجرا می‌کند.
- `.github/workflows/` برای lint، unit tests، contract tests با mocks، dependency/license/security scan و بررسی artifact manifests استفاده می‌شود. CI هرگز نباید از API key واقعی، secret VPN/SSH یا حساب واقعی صرافی استفاده کند.
- داده‌های تاریخی یا خبری دارای محدودیت بازتوزیع در Git commit نمی‌شوند؛ از synthetic/sanitized fixtures یا manifest و hash استفاده شود. وزن‌های بزرگ مدل نیز به‌جای Git در artifact store خصوصی با hash/revision نگهداری شوند.
- `docs/data-provider-registry.md` مرجع وضعیت مجوز و endpoint providerهاست؛ `docs/connectivity-test-matrix.md` مرجع transportها و `docs/phase-reports/phase-XX.md` شواهد پذیرش هر مرحله است.
- دستورهای تغییردهندهٔ مالک در `docs/governance/owner-directives/` یا GitHub Issue ثبت می‌شوند؛ هر directive دارای اثر امنیتی/مالی، آزمون و commit SHA است. متن دستور بدون merge/versioning مجوز دورزدن runtime gate نیست.
- گزارش‌بندی در `docs/reporting/` نگهداری می‌شود؛ هرگز Bale bot token، chat ID خصوصی یا دادهٔ حساب حساس را در GitHub Actions, issues, artifacts یا commit نگذارید.

## انتقال به سیستم محلی برای آزمون و live
1. از tag/release مشخص و commit تأییدشدهٔ `main` روی ماشین محلی یا سرور تحت کنترل مالک checkout کنید؛ فایل lock و hashها را تطبیق دهید.
2. secretهای تست را از secret manager محلی تزریق کنید؛ `config` دارای secret در Git نباشد. ابتدا محیط mock/sandbox، سپس dry-run روی live market data اجرا شود.
3. اجرای زنده تنها روی محیط جداگانه، با کلید API مختص همان worker و بدون مجوز برداشت فعال شود. GitHub Action نباید مسیر deploy مستقیم به سرمایهٔ واقعی داشته باشد.
4. قبل از live، release record شامل commit SHA، config/model hashes، provider status، contract/security/recovery test evidence، سقف محافظتی مطلق و امضای GO باید تولید شود.
5. به‌روزرسانی نسخهٔ live دستی/کنترل‌شده و قابل rollback باشد؛ retraining یا merge کد به‌تنهایی نسخهٔ live را تغییر ندهد.

## معیار پذیرش جریان GitHub
- [ ] branch protection و PR templates فعال‌اند؛ هیچ تغییر مستقیم و بازبینی‌نشده‌ای به `main` نمی‌رود.
- [ ] CI فقط mocks/sanitized data دارد و هیچ credential واقعی در logs/artifacts نیست.
- [ ] هر فاز دارای Issue، پذیرش قابل آزمون، گزارش نتیجه و GO/NO-GO است.
- [ ] یک release candidate از commit ثابت روی محیط محلی قابل بازتولید است.
- [ ] اجرای زنده از GitHub-hosted runner غیرممکن و به‌صورت فنی/عملیاتی ممنوع است.

# نمای کلی فازها
| فاز | عنوان | خروجی کلیدی | اجازهٔ معامله |
|---|---|---|---|
| ۰ | حاکمیت، کشف و خط مبنا | ADRها، نسخه‌ها، risk policy، ممیزی Novaryn و license inventory | خیر |
| ۱ | بستر پایه Freqtrade، اتصال و اعلان | محیط بازتولیدپذیر، ماتریس اتصال، Bale adapter و گزارش/نمودار آزمایشی با دادهٔ ساختگی | خیر؛ فقط fixture/backtest |
| ۲ | ادغام آداپتور نوبیتکس | contract adapter و test suite Spot | خیر؛ ابتدا mock/test environment |
| ۳ | Data Quality Gate | گیت بازار/ویژگی/خبر با reason code | خیر |
| ۴ | FreqAI baseline | مدل پیش‌بینی پایه و گزارش walk-forward | خیر |
| ۵ | Triple-Barrier و Meta-Labeling | DeepAlpha PoC یا Plan B سفارشی | خیر |
| ۶ | External Data، NLP و Social Media | provider adapters، News Store، local inference و مجوز داده/مدل | خیر |
| ۷ | Risk Engine و Kill Switch | gate حساب، supervisor و fault injection | خیر |
| ۸ | Multi-user / Multi-exchange foundation | control-plane schema و worker isolation | خیر؛ چند حساب غیرواقعی/آزمایشی |
| ۹ | یکپارچه‌سازی و backtest نهایی | گزارش end-to-end، stress test و model card | خیر |
| ۱۰ | Dry-run / paper | مشاهدهٔ واقعی بدون سرمایه، عملیات و reconciliation | خیر |
| ۱۱ | اجرای زندهٔ محدود | release امضاشده، سقف سرمایه و runbook فعال | بله، فقط با مجوز کتبی |
| ۱۲ | پایش، بازآموزی و توسعه | model promotion، rollback و اضافه‌کردن بازار/صرافی | مطابق gate جداگانه |

# فاز ۰ — حاکمیت، کشف و تعریف قراردادها
## هدف
پیش از پیاده‌سازی، تصمیم‌های مبهم به سیاست‌های قابل‌آزمون تبدیل شوند و ریسک‌های وابستگی و حقوقی معلوم گردند.

## گام‌ها
1. تثبیت نسخهٔ هدف Freqtrade، Python، CCXT، سیستم‌عامل/کانتینر و روش نصب؛ ذخیرهٔ commit/version و dependency lock.
2. دریافت و ثبت repository URL و commit دقیق آداپتور Novaryn؛ تعیین مالکیت، مجوز، وجود secret، کیفیت تست و نوع پیاده‌سازی آن (CCXT-compatible یا REST wrapper).
3. دریافت از مالک فقط «سقف محافظتی مطلق سرمایه» با مقدار، مرجع محاسبه، semantics فعال‌شدن و رفتار پس از halt؛ برای مثال حد نهایی افت equity یا trigger مشخص Kill Switch. در فاز صفر از مالک عدد ثابت برای daily risk budget، stake، position size، ATR multiplier یا stop-loss درخواست نشود؛ این موارد خروجی Dynamic Risk Allocation هستند. قواعد فنی صرافی/موجودی و ممنوعیت leverage همچنان قید اجرایی‌اند.
4. تعریف account/tenant identity، سیاست secret، جداسازی worker، retention داده و دسترسی‌های لاگ.
5. تهیهٔ inventory اتصال‌ها: Direct، HTTP/HTTPS Proxy، SOCKS Proxy، VPN و SSH Tunnel؛ routeهای مجاز، egress IP پایدار برای allowlist، health checks و محدودیت‌های شبکه ثبت شوند.
6. ایجاد provider registry برای داده‌های بازار، آنچین/نهنگ، خبر، تقویم اقتصادی، X، Telegram و منابع فارسی؛ API رسمی، پلن/rate limit، مجوز پردازش/retention و fallback ثبت شود.
7. تهیهٔ inventory مجوز برای DeepAlpha، dependencies، مدل‌های NLP، datasetها و providers خبر. TradingView تا اخذ مجوز صریح برای non-display/algorithmic use در تصمیم معامله blocked است.
8. تعیین جفت‌های USDT نسخهٔ اول و احراز اینکه هر صرافی هدف واقعاً همان بازارها را ارائه می‌کند.
9. تعریف معیارهای آماری پیش از اجرای backtest: معیار سود خالص بعد از هزینه، max drawdown، حداقل تعداد معاملات، پایداری بین foldها و شرایط رد مدل.
10. تصویب `Project Governance & Owner Authority`: فرمت Owner Directive، حدود change-control، Release Authorization، مسئول ثبت اثر مالی/امنیتی و تعارض با gateهای runtime.
11. تعیین `reporting.timezone` (پیش‌فرض پیشنهادی `Asia/Tehran` و قابل‌تغییر)، زمان ارسال ثابت ۰۷:۰۰، chat ID allowlist و سیاست محرمانگی گزارش Bale؛ token واقعی وارد مستند یا Git نشود.

## معیار پذیرش فاز ۰
- [ ] نسخه‌ها و dependencies دقیق و قابل بازتولید ثبت شده‌اند.
- [ ] URL، commit و مجوز آداپتور Novaryn ثبت شده و کد از نظر hard-coded secrets بررسی شده است.
- [ ] route policy و provider registry اولیه شامل endpoint رسمی، نرخ مجاز، مجوز پردازش، retention و fallback است.
- [ ] TradingView برای استفادهٔ خودکار non-display مسدود است مگر مجوز صریح و قابل استناد ثبت شود.
- [ ] تنها ورودی مالی الزام‌آور مالک، سقف محافظتی مطلق و تعریف دقیق اندازه‌گیری/زمان‌بندی/رفتار پس از trigger است؛ برای اندازهٔ پوزیشن، stop-loss و بودجهٔ ریسک روزانه عدد ثابت از مالک درخواست نشده است.
- [ ] مدل‌ها و خبر دارای license inventory و وضعیت `approved / conditional / rejected` هستند.
- [ ] فهرست جفت‌ها فقط شامل بازارهای احرازشده است.
- [ ] تعریف GO/NO-GO برای هر فاز مورد تأیید مالک پروژه است.
- [ ] سلسله‌مراتب Owner Directive و روش نسخه‌بندی/ممیزی آن در MRD و قالب GitHub ثبت شده است.
- [ ] منطقهٔ زمانی گزارش ۰۷:۰۰، گیرندهٔ مجاز Bale و طبقه‌بندی داده‌های مجاز در گزارش مشخص شده‌اند؛ هیچ credential در repo نیست.

**NO-GO اگر:** سقف محافظتی مطلق و رفتار Kill Switch مبهم یا آزمون‌ناپذیر است، آداپتور موجود قابل بررسی نیست، یا مجوزهای وابستگی‌های ضروری مبهم هستند.

# فاز ۱ — بستر پایه Freqtrade و Connectivity Manager
## هدف
ایجاد پایه‌ای قابل بازتولید بدون سیگنال AI زنده و بدون دسترسی به سرمایهٔ واقعی.

## گام‌ها
1. اجرای Freqtrade در محیط تمیز و version-pinned؛ جداکردن configهای test/dry-run/live template.
2. ساخت اسکلت strategy و مسیر FreqAI طبق قرارداد نسخهٔ مورد استفاده، بدون ورود live.
3. ساخت worker template با volume اختصاصی برای DB، `user_data`, models, logs و backtest results.
4. افزودن health checks، structured logging، graceful shutdown و شناسهٔ correlation برای runها.
5. نوشتن smoke test برای start/stop، load config، load strategy، load model placeholder و restart.
6. ساخت اسکلت `connectivity/manager/`, `connectivity/proxy/`, `connectivity/vpn/` و `connectivity/tunnel/` با route policy نسخه‌بندی‌شده؛ secrets فقط از secret mount خوانده شوند.
7. تست Direct، HTTP/HTTPS Proxy، SOCKS Proxy، VPN و SSH Tunnel روی endpointهای مجاز: DNS/TLS، egress IP، authentication، latency، reconnect و REST/WebSocket در صورت استفاده.
8. تست route health و failover فقط میان مسیرهای از پیش تأییدشده؛ برای order timeout تغییر مسیر/ارسال مجدد تا reconciliation ممنوع است.
9. ایجاد اسکلت `services/reporting/{scheduler,snapshots,charts,templates}/` و `integrations/notifications/bale/` با interface ارسال مستقل از موتور معامله.
10. بررسی مستندات جاری Bale Bot API و انتخاب SDK پس از audit مجوز، dependencies و contract؛ توکن و chat ID فقط در secret/config محلی امن قرار گیرند.
11. ساخت گزارش آزمایشی از دادهٔ synthetic شامل PnL ساختگی واضحاً برچسب‌خورده، تعداد معامله و نمودار PNG؛ ارسال به chat آزمایشی مجاز و تست schedule روزانهٔ ۰۷:۰۰ در timezone پیکربندی‌شده. در این فاز هیچ PnL واقعی/معاملاتی ادعا نمی‌شود.

## معیار پذیرش فاز ۱
- [ ] نصب تمیز از lockfile بدون دستکاری دستی محیط موفق است.
- [ ] backtest نمونه بدون خطای schema اجرا می‌شود و نتیجه ذخیره می‌شود.
- [ ] دو worker آزمایشی هیچ فایل DB یا writable state مشترک ندارند.
- [ ] secret در repo/log/image وجود ندارد.
- [ ] restart test بدون خراب‌شدن DB یا artifact موفق است.
- [ ] ماتریس اتصال Direct/HTTP(S)/SOCKS/VPN/SSH برای روش‌های قابل پشتیبانی نتیجهٔ pass/fail دارد.
- [ ] egress IP مسیر معاملاتی با allowlist صرافی سازگار است؛ TLS validation فعال است و هیچ credential در log/repository دیده نمی‌شود.
- [ ] route failure به fail-closed می‌انجامد و order timeout باعث retry کور یا duplicate order نمی‌شود.
- [ ] Bale adapter می‌تواند پیام متنی و یک نمودار PNG آزمایشی را به گیرندهٔ allowlist شده بفرستد؛ آزمون mock/contract سبز است.
- [ ] Scheduler در ۰۷:۰۰ منطقهٔ زمانی تعیین‌شده، پس از restart یک بار اجرا می‌شود و گزارش تکراری بدون revision ایجاد نمی‌کند.
- [ ] خطای API، گیرندهٔ غیرمجاز و token نامعتبر آزمایش شده‌اند؛ هیچ secret در log، screenshot یا artifact دیده نمی‌شود.

# فاز ۲ — بازاستفاده و ادغام آداپتور Spot نوبیتکس
## هدف
استفاده از آداپتور Novaryn به‌عنوان asset موجود، با حداقل لایهٔ سازگاری لازم؛ بدون نوشتن دوبارهٔ client API و بدون موتور سفارش موازی.

## گام‌ها
1. آداپتور را با commit ثابت به‌صورت submodule یا پکیج داخلی pin کنید؛ تاریخچه و تست‌های فعلی آن را حفظ کنید.
2. مشخص کنید آداپتور در کدام لایه قرار دارد: CCXT exchange subclass، client REST مستقل، یا connector دیگری. این بررسی مبنای انتخاب مرز ادغام است.
3. قرارداد exchange را با endpointهای رسمی نوبیتکس تطبیق دهید: markets, balance, order add, order status, cancel, fills, fees, precision, min/max, rate limits, pagination و error handling.
4. مطمئن شوید همهٔ عملیات معامله از lifecycle Freqtrade عبور می‌کند. REST wrapper مستقیم نباید جداگانه سفارش ثبت کند.
5. اجرای contract tests هم از مسیر Direct و هم route نهایی Connectivity Manager؛ route switch بعد از timeout سفارش بدون reconciliation ممنوع است.
6. اگر exchange class سفارشی نیاز به تغییر در هستهٔ Freqtrade دارد، ابتدا upstream/extension path بررسی شود؛ تغییرات local core بدون ADR، CI و نسخهٔ pin شده ممنوع است.
7. تست‌ها را ابتدا روی fixture/mock و سپس test environment یا کوچک‌ترین محیط بدون سرمایهٔ واقعی اجرا کنید؛ اگر محیط تست واقعی وجود ندارد، سفارش واقعی برای اثبات اولیه مجاز نیست.
8. market symbols را از endpoint فهرست بازارها بخوانید و فقط جفت‌های USDT که واقعاً پشتیبانی می‌شوند به allowlist اضافه کنید.

## معیار پذیرش فاز ۲
- [ ] تمام contract tests شامل partial fill، cancel/status، retry، rate limit و unknown order سبز هستند.
- [ ] precision/minimums و mapping نمادها با API واقعی مطابقت دارند.
- [ ] تکرار درخواست پس از timeout منجر به سفارش تکراری نمی‌شود یا به reconciliation و توقف می‌رسد.
- [ ] Worker پس از restart open orders و balances را با صرافی reconcile می‌کند.
- [ ] یک مسیر واحد ثبت سفارش وجود دارد: Freqtrade → adapter → Nobitex.
- [ ] هیچ برداشت/withdraw permission در کلید مورد استفاده وجود ندارد.

**NO-GO اگر:** برای اتصال، به دو executor هم‌زمان یا patch پنهان در Freqtrade نیاز باشد؛ ابتدا ADR و مسیر upstream/fork کنترل‌شده تصویب گردد.

# فاز ۳ — Data Quality Gate
## هدف
تضمین این‌که دادهٔ ناسالم، stale، ناقص یا آلوده به leakage پیش از آموزش و inference رد یا با دلیل روشن قرنطینه شود.

## گام‌ها
1. تعریف schema نسخه‌بندی‌شده برای OHLCV، feature vector، خبر و خروجی مدل.
2. پیاده‌سازی/تست کنترل timestamp، timezone، duplicates، gap، OHLC invariants، NaN/inf، volume و stale thresholds.
3. ایجاد gap repair با دریافت مجدد داده؛ از interpolation پنهان OHLCV خودداری شود.
4. تعریف قرارداد دادهٔ خارجی شامل source/provider ID، event/ingested/available timestamps، units، quote currency، revision، license status، rate limit و provenance.
5. اعتبارسنجی timestamp/pagination/gap/schema/freshness برای analytics و on-chain؛ دادهٔ provider ثالث جایگزین قیمت اجرای صرافی نیست.
6. تعریف `VALID/DEGRADED_ALLOWED/REJECTED` و reason codeهای استاندارد.
7. ایجاد تست‌های synthetic data و regression dataset برای هر خرابی شناخته‌شده.
8. اتصال gate به مسیر پیش از feature engineering و پیش از inference؛ ثبت fingerprint/window در ledger.

## معیار پذیرش فاز ۳
- [ ] هر fault تزریق‌شده به reason code درست منجر می‌شود.
- [ ] stale data و gap ورود جدید را متوقف می‌کنند.
- [ ] fixtureهای سالم کاذب رد نمی‌شوند و مرزهای tolerance مستندند.
- [ ] هیچ پرکردن خودکار OHLCV بدون policy/test وجود ندارد.
- [ ] backtest و live از قواعد اعتبارسنجی هم‌معنا استفاده می‌کنند.
- [ ] هر provider خارجی status مجوز، rate limit، retention و policy تازگی مشخص دارد؛ source ممنوع/منقضی به مدل تزریق نمی‌شود.

# فاز ۴ — خط مبنای FreqAI و ارزیابی کمی
## هدف
ایجاد baseline قابل تکرار پیش از استفاده از DeepAlpha، Meta-Labeling و NLP؛ تا بهبودها قابل سنجش باشند.

## گام‌ها
1. تعریف feature/target contract، افق پیش‌بینی، هزینهٔ معامله، fee/slippage model و معیارهای پذیرش.
2. پیاده‌سازی مدل پایه در `user_data/freqaimodels/` با `IFreqaiModel` نسخهٔ pinشده و feature engineering مبتنی بر دادهٔ گذشته.
3. اجرای walk-forward زمانی و OOS؛ leakage detection، stability per pair/regime و sensitivity analysis.
4. ثبت مدل/feature schema/params/metrics در manifest؛ مقایسه با baseline ساده و معیار بدون معامله.
5. ارزیابی هزینهٔ محاسبات، train duration، prediction latency و رفتار model reload.
6. آماده‌سازی خروجی‌های نسخه‌بندی‌شده برای Dynamic Risk Allocation (regime probabilities, ATR/volatility, calibrated confidence/uncertainty, expected edge net of costs)؛ در این فاز فقط ثبت/ارزیابی offline، نه تخصیص سرمایهٔ زنده.

## معیار پذیرش فاز ۴
- [ ] مدل قابل reload و predict با artifactهای همان نسخه است.
- [ ] split زمانی، gap/embargo در صورت لزوم و هزینه‌های معامله مستند هستند.
- [ ] گزارش OOS شامل max drawdown، net return، turnover، fees، تعداد معاملات و پایداری fold است.
- [ ] نتایج با همان commit/config/dataset دوباره قابل تولیدند.
- [ ] مدل حداقل شرایط از پیش تعریف‌شده را در چند پنجره می‌گذراند؛ در غیر این صورت مدل رد می‌شود و هیچ promotion رخ نمی‌دهد.
- [ ] confidence/uncertainty و خروجی regime در بازه‌های زمانی خارج از نمونه ارزیابی شده‌اند؛ confidence خام بدون calibration برای sizing پذیرفته نیست.

# فاز ۵ — Triple-Barrier و Meta-Labeling
## هدف
افزودن برچسب‌گذاری بر اساس مسیر قیمت و مدل کیفیت فرصت بدون وابستگی قفل‌شده به DeepAlpha.

## گام‌ها
1. PoC ایزولهٔ DeepAlpha: source audit، license audit، install/reload، سازگاری `IFreqaiModel`، train/predict/backtest، no-core-patch.
2. مقایسهٔ مستندات PyPI/GitHub با رفتار واقعی همان revision و dependency graph.
3. اگر PoC رد شد یا هزینهٔ نگهداری بالا بود، Triple-Barrier target و Composite FreqAI model طبق MRD ساخته شود.
4. primary signalها برای آموزش meta-model باید out-of-fold تولید شوند؛ سیگنال‌های in-sample رد می‌شوند.
5. اجرای purged walk-forward و embargo؛ محاسبهٔ outcome با fees/slippage و trade exit contract.
6. تست سازگاری ذخیره/بازیابی primary و meta artifactها و versioned feature schema.

## معیار پذیرش فاز ۵
- [ ] انتخاب DeepAlpha یا Plan B با ADR و شواهد ثبت‌شده نهایی شده است.
- [ ] برچسب Triple-Barrier هیچ feature آینده‌ای را وارد ورودی مدل نمی‌کند.
- [ ] Meta-Label روی OOF signalها آموزش دیده و leakage test دارد.
- [ ] عملکرد در برابر baseline، در چند fold و با هزینه، گزارش شده است.
- [ ] شکست artifact/MetaLabel باعث no-entry می‌شود.
- [ ] هیچ فایل داخلی Freqtrade به‌صورت پنهانی patch نشده است.

# فاز ۶ — External Data، NLP و Social Media
## هدف
افزودن داده‌های خارجی و اجتماعی به feature pipeline با اتصال پایدار، مجوز قابل دفاع و NLP محلی؛ دادهٔ زودتر در دسترس، پس از کنترل کیفیت، می‌تواند سریع‌تر به ویژگی تبدیل شود، اما کیفیت پیش‌بینی تضمین‌شده نیست.

## منابع کاندید
- بازار/تحلیل: TradingView، CoinGecko، CoinMarketCap، CoinGlass.
- آنچین/نهنگ‌ها: On-Chain Data از APIهای مجاز، Whale Alert.
- خبر/رویداد: RSS News، Economic Calendar و APIهای خبری دارای مجوز.
- شبکه‌های اجتماعی: X/Twitter API، Telegram API/کانال‌های مجاز و سایر Social Media APIها.
- منابع ایرانی: سایت‌های تحلیلی و خبری فارسی از طریق API رسمی/RSS مجاز.

**محدودیت TradingView:** تا اخذ مجوز صریح برای non-display/algorithmic use، دادهٔ آن برای ورودی NLP/Feature و تصمیم معامله blocked است. از scraping، API غیررسمی یا cookies برای دورزدن محدودیت استفاده نشود.

## گام‌ها
1. ثبت هر provider در `docs/data-provider-registry.md`: API رسمی، شرایط استفاده، پلن و rate limits، retention/redistribution، timestamp semantics، وضعیت مجوز و owner؛ شروع با حداقل یک feed خبری و یک منبع market analytics مجاز.
2. ingest دادهٔ عددی بازار/analytics به External Feature Store؛ ingest RSS/news/calendar/social مجاز به News Store؛ سپس NLP Service فقط متن‌های مجاز را پردازش کند.
3. ساخت event schema شامل `published_at`, `ingested_at`, `available_at`, `processed_at`, source, asset mapping, content hash و version.
4. benchmark مدل‌های NLP candidate روی مجموعهٔ برچسب‌خوردهٔ اختصاصی از خبرهای رمزارز؛ بررسی language/domain shift و confidence calibration.
5. freeze و pin کردن model revision، tokenizer، dependency lock و checksum وزن‌ها.
6. اجرای inference batch/offline برای backtest؛ snapshot تاریخی خبر و زمان واقعی در دسترس بودن حفظ شود.
7. اجرای shadow inference زنده بدون اثر بر سفارش‌ها و مقایسهٔ latency/error rates؛ خبر/social features فقط پس از ablation و گذر از Data Quality Gate بررسی شوند.

## معیار پذیرش فاز ۶
- [ ] license manifest برای وزن، dataset و source تکمیل و review شده است.
- [ ] تمامی منابع فعال از API/RSS مجاز یا مجوز قراردادی معتبر استفاده می‌کنند و scraping ممنوع انجام نمی‌شود.
- [ ] TradingView برای AI/non-display trading غیرفعال است مگر مجوز صریح ثبت شده باشد.
- [ ] دادهٔ عددی بازار به External Feature Store و متن خبر/social مجاز به News Store/NLP Service می‌رود؛ هر رکورد source/version/timestamps دارد.
- [ ] latency/queue lag/freshness و مجوز fallback مسیر در monitoring ثبت می‌شوند.
- [ ] خبر بعد از decision time در backtest هرگز به feature گذشته وارد نمی‌شود.
- [ ] duplicate/news mapping/timestamp tests سبز هستند.
- [ ] NLP timeout/down رفتار fail-closed مطابق policy دارد.
- [ ] benchmark هر مدل شامل F1 per class, calibration, latency و هزینهٔ حافظه است.
- [ ] اثر NLP در backtest ablation نسبت به مدل بازار بدون خبر اندازه‌گیری شده است؛ عدم بهبود یعنی حذف یا محدودکردن آن.

# فاز ۷ — Dynamic Risk Allocation، Risk Gate و Kill Switch
## هدف
اثبات اینکه هوش مصنوعی می‌تواند بودجهٔ روزانه، اندازهٔ موقعیت و stop-loss را در واکنش به regime/ATR/confidence به‌طور پویا تنظیم کند، درحالی‌که سقف محافظتی مطلق مستقل و غیرقابل‌تخطی باقی می‌ماند.

## گام‌ها
1. تثبیت source of truth برای equity، realized/unrealized PnL، fees، open orders و محاسبهٔ سقف محافظتی مطلق؛ این گام سیاست owner را پیاده می‌کند و عددهای پویای risk allocation را hard-code نمی‌کند.
2. تعریف قرارداد خروجی Dynamic Risk Allocation: risk budget پیشنهادی روزانه/معامله، stake، stop price/distance، regime، ATR snapshot، confidence کالیبره‌شده، uncertainty، expected edge پس از هزینه‌ها و model revision.
3. پیاده‌سازی allocator که نوسان/ATR، رژیم، uncertainty/calibration، هزینه‌ها، equity باقی‌مانده، exposure و correlation را در تصمیم لحاظ کند؛ تغییر شرایط باید بتواند risk budget را کاهش دهد یا به no-trade برسد.
4. پیاده‌سازی Risk Gate مستقل برای بازحساب زیان تخمینی در stop، fees/slippage، موجودی آزاد، precision بازار و ظرفیت باقی‌مانده تا سقف مطلق؛ پیشنهاد معتبر مدل خودبه‌خود order permission محسوب نمی‌شود.
5. افزودن supervisor مستقل برای پایش سقف مطلق، latched halt و رفتار recovery؛ AI به policy store و Kill Switch دسترسی نوشتن ندارد.
6. fault injection: سقف مطلق hit، stop نامعتبر، confidence/calibration degrade، ATR spike، regime shift، risk service down، stale price، API timeout، supervisor crash، restart و وضعیت سفارش نامعلوم.
7. نوشتن runbook برای halt، pending order reconciliation، کنترل موقعیت‌های باز و re-arm دستی.

## معیار پذیرش فاز ۷
- [ ] AI/NLP process هیچ توانایی نوشتن در risk policy یا Kill Switch ندارد.
- [ ] تست‌های قطعی نشان می‌دهند با افزایش volatility/ATR یا uncertainty و افت calibration، اندازه کاهش می‌یابد یا معامله رد می‌شود؛ تغییر اندازه در شرایط متفاوت قابل مشاهده و قابل بازتولید است.
- [ ] stake پیشنهادی با فاصلهٔ stop، fees و slippage سازگار است و از موجودی آزاد/قواعد بازار عبور نمی‌کند.
- [ ] سقف محافظتی مطلق حتی با confidence بسیار بالا، مدل خراب یا restart دور زده نمی‌شود؛ Kill Switch به‌طور خودکار reset نمی‌شود.
- [ ] نبود داده یا خروجی allocator، stop نامعتبر و failure در Risk Gate به عدم ورود می‌انجامد.
- [ ] مقایسهٔ walk-forward/OOS با baseline ساده، drawdown، tail loss/CVaR، هزینه، گردش معاملات و پایداری بین رژیم‌ها را گزارش می‌کند.
- [ ] خروج‌ها و اقدامات لازم برای کاهش ریسک به‌طور غیرعمد توسط news/risk-entry gate مسدود نمی‌شوند.
- [ ] fault injection گزارش قابل ممیزی و نتیجهٔ pass/fail دارد.

# فاز ۸ — پایهٔ Multi-User / Multi-Account / Multi-Exchange
## هدف
پیاده‌سازی مرزهای tenant و worker isolation قبل از اضافه‌شدن کاربران واقعی.

## گام‌ها
1. طراحی schema برای tenant/account/exchange/environment/strategy version و stateهای lifecycle.
2. ساخت template ایجاد worker، volume، database و log جداگانه؛ از ذخیرهٔ secret در template اجتناب شود.
3. یک secret reference جدا برای هر account/environment و مجوز API حداقلی.
4. کنترل‌های ریسک در سطح حساب و supervisor تجمیعی پرتفوی؛ تعریف رفتار هنگام از دست رفتن control plane.
5. افزودن account-level alerting، access control، audit log و تست cross-tenant access denial.
6. تست دو worker هم‌زمان با حساب‌های fake/mock برای اثبات عدم اشتراک writable state.

## معیار پذیرش فاز ۸
- [ ] tenant A به DB, logs, secrets, ledger یا model cache tenant B دسترسی ندارد.
- [ ] rollout/rollback یک worker حساب دیگر را تغییر نمی‌دهد.
- [ ] portfolio supervisor می‌تواند ورود همهٔ حساب‌های مشمول را در رسیدن به cap مسدود کند.
- [ ] از duplicate worker فعال برای یک account/exchange/environment جلوگیری می‌شود.
- [ ] افزودن connector جدید نیازمند تغییر در core trading logic نیست و contract tests را می‌گذراند.

# فاز ۹ — یکپارچه‌سازی end-to-end و آزمون نهایی
## هدف
اثبات زنجیرهٔ کامل از داده تا رأی ریسک، معاملهٔ شبیه‌سازی‌شده و Ledger.

## گام‌ها
1. اجرای سناریوهای ورود، عدم ورود، خروج، stop-loss پویا، خبر stale، خبر conflicting، مدل unavailable، gap، market halt و order timeout.
2. اجرای سناریوهای رژیم کم/پُرنوسان، ATR spike، اطمینان پایین/بدکالیبره و همبستگی بالا؛ اثبات شود Dynamic Risk Allocation اندازه/stop/budget را تغییر می‌دهد یا معامله را حذف می‌کند، بدون تغییر سقف محافظتی مطلق.
3. اجرای backtestهای نهایی با هزینه و معیارهای مصوب و ثبت hash تمام inputs.
4. تست recovery در هر نقطهٔ بحرانی: پیش از سفارش، پس از ارسال و پیش از دریافت status، partial fill، cancel و restart.
5. تست ظرفیت: prediction latency، news queue lag، DB write throughput، worker restart time و API rate limits.
6. ممیزی امنیت، دسترسی secret، file permissions، dependency vulnerabilities، logs و retention.
7. پیاده‌سازی Reporting Engine روی دادهٔ واقعی Freqtrade DB و Decision Ledger، شامل snapshot reconcileشده، محاسبهٔ realized/unrealized/net PnL، trade counts، equity/drawdown، open orders، exposure، Risk Gate/Kill Switch، Data Quality و وضعیت مدل/اتصال.
8. تولید نمودارهای گرافیکی نسخه‌بندی‌شده (equity/PnL/trades/risk/health) و قالب خلاصهٔ روزانهٔ قابل خواندن روی موبایل؛ گزارش فقط دادهٔ قابل‌اعتماد را به‌عنوان قطعی نشان می‌دهد.
9. ثبت `report_id`, `data_cutoff`, snapshot hash, generated_at, delivery status؛ افزودن retry محدود، idempotency، fallback متنی و redaction اطلاعات حساس.
10. تولید release candidate و checklist امضای معماری/Quant/Risk/Operations.

## معیار پذیرش فاز ۹
- [ ] تمام end-to-end tests و contract tests در CI سبز هستند.
- [ ] order timeout بدون reconciliation retry کور ایجاد نمی‌کند.
- [ ] از یک `decision_id` می‌توان ورودی، خروجی مدل، رأی ریسک و سفارش را بازسازی کرد.
- [ ] گزارش عملکرد خارج از نمونه و محدودیت‌های مدل برای بازبین‌ها آماده است.
- [ ] خروجی dynamic sizing/stop در چند رژیم از نظر drawdown/tail risk/fees ارزیابی شده و شکست آن به no-go می‌انجامد.
- [ ] هیچ finding با severity بالا بدون mitigation یا پذیرش ریسک رسمی باقی نمانده است.
- [ ] PnL و تعداد معاملات گزارش با source of truth/صرافی reconcile شده‌اند و mismatch گزارش را با برچسب DEGRADED می‌کند.
- [ ] نمودارها، timezone، snapshot hashes، redaction، retries و idempotency دارای tests و evidence هستند.

# فاز ۱۰ — Dry-run / Paper در بازار زنده
## هدف
مشاهدهٔ رفتار عملیاتی در شرایط بازار جاری، بدون سرمایهٔ واقعی؛ زمان دوره باید از پیش تعیین و متناسب با تعداد فرصت‌ها باشد، نه صرفاً چند روز تقویمی.

## گام‌ها
1. اجرای worker با live data و `dry_run`؛ هیچ permission معاملهٔ واقعی فعال نباشد.
2. پایش Data Quality, model inference, risk rejections, simulated fills, latency, rate limits, restart و alerting.
3. مقایسهٔ سیگنال‌ها با قیمت‌های تحقق‌یافته و محاسبهٔ slippage حساس به فرض fill.
4. اجرای drill قطع شبکه، API rate limiting، stale candles و kill switch.
5. تثبیت runbook و هزینهٔ عملیات؛ ثبت تمام انحرافات و رفع آن‌ها قبل از promotion.
6. اجرای برنامهٔ گزارش روزانه در dry-run/paper در ۰۷:۰۰ timezone پیکربندی‌شده طی دورهٔ مصوب؛ مقایسهٔ خروجی با Freqtrade DB، Decision Ledger و snapshot حساب.
7. تزریق خرابی Bale API/network، stale data، mismatch در reconciliation و restart scheduler؛ تأیید alert/reties/fallback و جلوگیری از گزارش قطعی گمراه‌کننده.

## معیار پذیرش فاز ۱۰
- [ ] دورهٔ مصوب بدون regression بحرانی سپری شده است.
- [ ] نرخ stale/gap و خطای سرویس در محدودهٔ تعریف‌شده قرار دارد.
- [ ] kill switch و recovery drill در محیط paper موفق بوده است.
- [ ] تصمیمات reject/allow توضیح‌پذیرند و Ledger کامل است.
- [ ] تخمین‌ها پس از لحاظ هزینه و slippage همچنان معیارهای از پیش تصویب‌شده را می‌گذرانند.
- [ ] گزارش‌های روزانه در ساعت تعیین‌شده تولید شده‌اند؛ ارقام با source of truth تطبیق دارند یا به‌روشنی DEGRADED علامت خورده‌اند.
- [ ] ارسال به Bale allowlist شده، retry/fallback و recovery آزموده شده‌اند و هیچ secret/اطلاعات حساس افشا نشده است.

# فاز ۱۱ — اجرای زنده با سرمایهٔ محدود
## پیش‌شرط غیرقابل‌چشم‌پوشی
اجرای زنده فقط پس از امضای جداگانهٔ **Release Authorization** انجام می‌شود. این سند به‌تنهایی مجوز مالی محسوب نمی‌شود.

## گام‌ها
1. تصویب کتبی سرمایهٔ شروع و سقف محافظتی مطلق، انتخاب جفت‌های مجاز و semantics توقف/ازسرگیری؛ از مالک عدد ثابت برای daily risk budget، stake، position size یا stop-loss درخواست نمی‌شود.
2. ایجاد API key جدید مخصوص live worker؛ بدون withdrawal، با IP restriction در صورت پشتیبانی؛ dry-run keys دوباره استفاده نشوند.
3. deploy مدل/config immutable با hash ثبت‌شده؛ ابتدا یک حساب، یک صرافی و allowlist کوچک.
4. اجرای smoke test غیرمعاملاتی/حداقل ریسک طبق امکانات venue و بازبینی balance/open orders قبل از شروع.
5. نظارت تقویت‌شده بر اولین دورهٔ معاملات و اختیار halt مستقل از توسعه‌دهنده/مدل.
6. پس از هر ناهنجاری مهم، توقف ورود، reconciliation و RCA قبل از ادامه.

## معیار پذیرش فاز ۱۱
- [ ] همهٔ release gates امضا شده‌اند.
- [ ] سرمایهٔ شروع و سقف محافظتی مطلق در policy version مشخص‌اند؛ پارامترهای روزانه/معامله‌ای از allocator پویا می‌آیند.
- [ ] API key محدود، بدون برداشت و مختص worker است.
- [ ] supervisor، alert و runbook آماده و آزمایش شده‌اند.
- [ ] ترتیب rollback/stop، مدیریت orderهای باز و بازیابی پس از crash مکتوب است.
- [ ] هیچ deploy خودکار مدل یا config بدون approval در live فعال نیست.
- [ ] Release Authorization توسط مالک ثبت شده و شامل commit/config/model hashes، risk/security impact و rollback است؛ درخواست مالک اجرا را آغاز می‌کند اما از gate شکست‌خورده عبور نمی‌کند.
- [ ] گزارش گرافیکی روزانهٔ ۰۷:۰۰ در Bale برای chat تأییدشده ارسال می‌شود؛ نتیجهٔ تحویل پایش و ثبت می‌شود.

## معیار توقف خودکار در live
- رسیدن به سقف محافظتی مطلق یا فعال‌شدن Kill trigger مصوب؛
- اختلاف غیرقابل‌حل در balance/order/trade؛
- stale/gapped data خارج از tolerance؛
- مدل/feature/schema نامنطبق؛
- outage در supervisor/ledger یا جهش خطای صرافی؛
- احتمال duplicate order؛
- مشاهدهٔ رخداد امنیتی یا افشای secret.

در هر مورد، default اقدام توقف ورود جدید است؛ مدیریت موقعیت موجود طبق runbook انجام می‌شود.

# فاز ۱۲ — عملیات، بازآموزی کنترل‌شده و گسترش بازار
## هدف
افزایش تدریجی قابلیت‌ها پس از اثبات ثبات نسخهٔ اول.

## گام‌ها
1. راه‌اندازی drift monitoring و بررسی دوره‌ای calibration/feature distribution.
2. retrain در محیط research بر اساس data snapshot نسخه‌بندی‌شده؛ نتایج با champion/challenger مقایسه شوند.
3. ارتقای مدل فقط پس از walk-forward/OOS، model card، risk review و paper/shadow period.
4. اضافه‌کردن provider خارجی یا social source جدید با adapter مستقل، ثبت مجوز/rate limits، contract/freshness tests و ablation قبل از فعال‌سازی.
5. اضافه‌کردن جفت، ارز مظنه یا exchange جدید با یک ADR و اجرای دوبارهٔ contract/security/recovery tests.
6. گسترش به چند کاربر با quota، isolation، audit، account-level and portfolio-level limits و تمرین incident response.

## معیار پذیرش فاز ۱۲
- [ ] هیچ مدل خودآموزی مستقیماً live model را overwrite نمی‌کند.
- [ ] rollback به champion قبلی آزموده شده است.
- [ ] connector/market جدید همهٔ contract tests را می‌گذراند.
- [ ] گزارش عملکرد و ریسک به تفکیک حساب/صرافی در دسترس است.
- [ ] تغییرات بازار/feature schema مستلزم migration و approval است.

# معیارهای کیفیت مشترک
| حوزه | معیار پذیرش |
|---|---|
| بازتولیدپذیری | هر نتیجه با revisionها، config، داده و seedهای ثبت‌شده قابل تکرار باشد. |
| جلوگیری از leakage | point-in-time join، feature/target separation و purged walk-forward آزموده باشند. |
| کنترل مالی | سقف محافظتی مطلق مستقل از مدل، fail-closed و دارای تست fault injection باشد؛ بودجهٔ روزانه/اندازه/stop پویا و OOS-validated باشد. |
| چرخهٔ سفارش | timeout/retry/cancel/partial fill/restart بدون وضعیت دوگانه یا سفارش تکراری آزموده شوند. |
| امنیت چندمستاجری | جداسازی secret، DB، log، model artifact و filesystem با تست منفی تأیید شود. |
| مجوز | مجوز نرم‌افزار، وزن مدل، dataset، منبع خبر و attribution در manifest ثبت شود. |
| توضیح‌پذیری عملیاتی | decision_id همهٔ ورودی‌ها، خروجی مدل، رأی ریسک و order outcome را به هم متصل کند. |
| rollout | live promotion و rollback فقط با release record و تایید مجاز باشند. |
| گزارش‌دهی | گزارش ۰۷:۰۰ قابل‌بازتولید، reconcile شده، با نمودار، report_id و delivery status؛ secrets هرگز ثبت/ارسال نمی‌شوند. |
| حاکمیت مالک | هر Owner Directive دارای نسخه، اثر ریسک/امنیت، آزمون، commit و تصمیم GO/NO-GO است. |

# قالب گزارش پایان هر فاز
هر فاز باید یک گزارش در `docs/phase-reports/phase-XX.md` داشته باشد و حداقل موارد زیر را ثبت کند:
- اهداف فاز و commitهای واردشده؛
- نسخهٔ ابزارها، datasets، model revisions و config hashes؛
- تست‌ها، نتایج دقیق و شواهد CI؛
- یافته‌های امنیتی/حقوقی و mitigation؛
- metricها و اختلاف با معیار پذیرش؛
- وضعیت نهایی `GO / NO-GO / CONDITIONAL GO`؛
- مسئول تصویب و تصمیم‌های نیازمند ADR.
- وضعیت گزارش روزانه: timezone، report_id نمونه، تطبیق PnL، وضعیت ارسال Bale و آزمون‌های خطا (بدون token/chat secret)
- Owner Directiveهای مؤثر، risk/security impact و commit SHA مربوط

# منابع اصلی
- Freqtrade FreqAI developer docs: https://www.freqtrade.io/en/stable/freqai-developers/
- FreqAI feature engineering: https://www.freqtrade.io/en/stable/freqai-feature-engineering/
- Freqtrade exchange docs: https://www.freqtrade.io/en/stable/exchanges/
- Freqtrade protections: https://www.freqtrade.io/en/stable/plugins/
- DeepAlpha PyPI: https://pypi.org/project/deepalpha-freqai/
- DeepAlpha repository: https://github.com/stefanoviana/deepalpha
- Nobitex API docs: https://apidocs.nobitex.ir/
- Bale Bot API / documentation entry point: https://docs.bale.ai/ (اعتبار و endpointهای جاری باید در فاز ۰/۱ بررسی شوند).
- baleRobot package page (third-party SDK candidate; not pre-approved): https://pypi.org/project/baleRobot/
- TradingView policies: https://www.tradingview.com/policies/
- CoinGecko API pricing/docs entry point: https://www.coingecko.com/en/api/pricing
- CoinMarketCap API docs: https://coinmarketcap.com/api/documentation/
- CoinGlass API docs: https://docs.coinglass.com/reference/endpoint-overview
- Whale Alert API docs: https://developer.whale-alert.io/api-account/documentation
- X API access policy: https://help.x.com/en/rules-and-policies/x-api
- Telegram API docs: https://core.telegram.org/api
- News API docs: https://newsapi.org/docs
- CCXT proxy usage: https://github.com/ccxt/ccxt/wiki/proxy-usage