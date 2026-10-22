# خلاصهٔ مدیریتی
AliX یک سامانهٔ معاملاتی **خودمختار از نظر تصمیم معاملاتی، محدودشده از نظر اختیار و قابل ممیزی از نظر عملیات** است. Freqtrade مسئول چرخهٔ سفارش، اتصال صرافی، پایگاه دادهٔ معاملات و بک‌تست باقی می‌ماند. FreqAI میزبان چرخهٔ آموزش و پیش‌بینی است. لایهٔ AliX، قرارداد ویژگی‌ها، کنترل کیفیت داده، تصمیم‌گیری چندمدلی، NLP، دفتر تصمیمات، جداسازی حساب‌ها و کنترل ریسک را اضافه می‌کند.

هشت اصلاح اساسی این نسخه:
- **آداپتور نوبیتکس Novaryn** دارایی موجود و نقطهٔ شروع است؛ بازنویسی از صفر مجاز نیست. تا زمان دریافت نشانی مخزن/commit و بررسی کد، اتصال مستقیم به Freqtrade تأییدشده محسوب نمی‌شود.
- **Multi-user / Multi-account / Multi-exchange** از ابتدا با نمونه‌های Freqtrade ایزوله، کنترل‌پلین مستقل و ناظر ریسک در سطح حساب و پرتفوی طراحی می‌شود.
- یک **Data Quality Gate مستقل و fail-closed** پیش از تولید ویژگی و پیش از ورود به مدل تعریف می‌شود.
- **Connectivity Manager** مسیرهای Direct، HTTP/HTTPS Proxy، SOCKS Proxy، VPN و SSH Tunnel را با policy و آزمون اتصال مدیریت می‌کند؛ این لایه برای دورزدن محدودیت‌ها نیست.
- **External Data & Social Media** به‌صورت adapterهای ماژولار، با registry حقوقی، API/RSS مجاز و مسیر جداگانهٔ News Store/NLP و External Feature Store اضافه می‌شود.
- در صورت ناسازگاری DeepAlpha، مسیر جایگزین متکی بر یک مدل ترکیبی سفارشی `IFreqaiModel`، برچسب‌گذاری Triple-Barrier و Meta-Labeling با پیش‌بینی‌های out-of-fold است.
- **Reporting Engine / Notification Service** گزارش روزانهٔ گرافیکی را ساعت ۰۷:۰۰ در منطقهٔ زمانی پیکربندی‌شده تولید و از طریق Bale Bot API برای شناسهٔ گفت‌وگوی مجاز مالک ارسال می‌کند؛ دقت ارقام، امنیت گیرنده و وضعیت ارسال ثبت می‌شوند.
- **حاکمیت مالک** صراحتاً تعریف شده است: مالک مرجع نهایی تصمیم‌های داخلی پروژه است و می‌تواند اسناد/سیاست‌ها را با تغییر نسخه‌بندی‌شده اصلاح کند؛ درخواست فعال‌سازی معاملهٔ زنده فقط از مسیر Release Authorization و پس از عبور گیت‌های اجباری اجرا می‌شود.

> وضعیت سند: معماری پیشنهادی برای تأیید. این سند هیچ ادعایی دربارهٔ سودآوری، سازگاری آزمایش‌نشدهٔ آداپتور، یا مجوز حقوقی نهایی مدل/داده مطرح نمی‌کند.

# کنترل سند و وضعیت تصمیم‌ها
| مورد | تصمیم / وضعیت |
|---|---|
| دامنهٔ نسخهٔ اول | Spot-only و جفت‌های دارای ارز مظنهٔ USDT؛ بازار هر صرافی باید از API واقعی احراز شود. |
| موتور معامله | Freqtrade؛ بدون بازنویسی order lifecycle، backtesting یا exchange sync. |
| مدل بازار | FreqAI با مدل سفارشی و feature/target contract نسخه‌بندی‌شده. |
| DeepAlpha | وابستگی اختیاری مشروط به ممیزی، سازگاری و تست عدم نشت داده. |
| NLP | مدل محلی Hugging Face، با pin نسخه/commit و ممیزی مجوز وزن و داده. |
| نوبیتکس | استفادهٔ مجدد از آداپتور Novaryn Spot؛ ادغام از طریق مرز سازگار با Freqtrade/CCXT. |
| اجرا | ابتدا تست، سپس backtest و dry-run، سپس سرمایهٔ محدود پس از تصویب release gate. |
| اصل امنیتی | مدل نمی‌تواند سیاست ریسک، کلیدها، پیکربندی انتشار یا وضعیت Kill Switch را تغییر دهد. |
| حاکمیت مالک | مالک بالاترین مرجع تصمیم‌گیری داخلی پروژه است؛ دستورات متعارض به change record نسخه‌بندی‌شده تبدیل می‌شوند، نه اجرای بی‌ردپا. |
| گزارش روزانه | گزارش گرافیکی در ۰۷:۰۰ منطقهٔ زمانی پیکربندی‌شده؛ تحویل از طریق Bale Bot API و ثبت نتیجهٔ ارسال. |

# ۱. چشم‌انداز و اهداف (Vision & Scope)
## ۱.۱ تعریف AliX
AliX یک سامانهٔ AI-driven برای انتخاب دارایی، تشخیص رژیم بازار، برآورد جهت/بازده و عدم‌قطعیت، تصمیم ورود/نگهداری/خروج، و پیشنهاد اندازهٔ معامله است. خودمختاری به معنی حذف دخالت روزمره در تصمیم‌های معمول معاملاتی است؛ نه اعطای دسترسی نامحدود به مدل. هر تصمیم از دروازهٔ قطعی کیفیت داده و کنترل ریسک عبور می‌کند و همهٔ ورودی‌ها و نسخه‌ها قابل بازسازی‌اند.

## ۱.۲ اهداف فنی
- تصمیم ENTRY/HOLD/EXIT عمدتاً از خروجی مدل‌های نسخه‌بندی‌شده ناشی شود؛ قواعد قطعی محدود به کیفیت داده، سازگارسازی، ایمنی و اجرای درست باشند.
- ترکیب مدل پیش‌بینی اصلی با Meta-Labeling برای برآورد کیفیت فرصت و اجتناب از اعتماد کور به یک خروجی خام.
- استفاده از Triple-Barrier برای برچسب‌گذاری نتایج بر اساس حد سود/حد ضرر/حد زمانی، با جلوگیری از نشت آینده به ویژگی‌ها.
- تبدیل خبر به ویژگی زمان‌مند با ثبت زمان انتشار، زمان دریافت و زمان دسترس‌پذیری واقعی.
- امکان افزودن صرافی، ارز مظنه یا مدل NLP بدون تغییر در سیاست ریسک و بدون بازنویسی موتور معاملات.
- داشتن مسیر سرتاسری برای توضیح تصمیم: داده → کنترل کیفیت → ویژگی → پیش‌بینی → پیشنهاد → رأی ریسک → سفارش/عدم سفارش → نتیجه.
- بازیابی پس از قطع شبکه، restart و crash بدون سفارش تکراری ناشی از عدم قطعیت وضعیت.

## ۱.۳ محدودهٔ نسخهٔ اول
**داخل محدوده:** معاملات Spot روی جفت‌های USDT؛ Freqtrade/FreqAI؛ مدل ML سفارشی؛ آزمایش DeepAlpha؛ سرویس NLP محلی؛ Data Quality Gate؛ Risk Supervisor؛ ثبت وقایع تصمیم و سفارش؛ جداسازی حساب‌ها و محیط‌ها در طراحی؛ آداپتور موجود نوبیتکس به‌عنوان مسیر یکپارچه‌سازی.

**خارج از محدوده:** Futures، leverage، margin، short، برداشت دارایی، ارسال سفارش مستقیم توسط LLM، تغییر خودکار کد یا سیاست ریسک، اشتراک کلید API بین کاربران، و rollout خودکار مدل جدید به سرمایهٔ واقعی بدون gate.

## ۱.۴ الزامات غیرقابل‌مذاکره
- Freqtrade منبع حقیقت lifecycle سفارش و معامله باقی می‌ماند.
- مدل AI، NLP و هر فرایند training از کلید API معاملاتی و دسترسی نوشتن به سیاست ریسک محروم است.
- در صورت قدیمی/ناقص بودن داده، نبود مدل معتبر، ناسازگاری نسخه، شکست سرویس ریسک یا ابهام وضعیت سفارش: **ورود جدید متوقف می‌شود**.
- خروج برای کاهش ریسک، مدیریت stop و همگام‌سازی وضعیت نباید به علت خطای یک سیگنال NLP مسدود شود.
- سودآوری، Sharpe یا win-rate از پیش تضمین نمی‌شود. پذیرش فقط بر مبنای ارزیابی زمان‌مند و خارج از نمونه انجام می‌شود.

## ۱.۵ حاکمیت پروژه و اختیارات مالک (Project Governance & Owner Authority)

**مالک پروژه مرجع نهایی تصمیم‌گیری داخلی AliX است.** هر سند، ADR، roadmap، قرارداد فنی یا قاعدهٔ داخلی با دستور صریح مالک قابل اصلاح یا جایگزینی است. در صورت تعارض، منظور مالک مبنای تصمیم و اصلاح سند خواهد بود؛ تغییر باید در GitHub Issue/Owner Directive ثبت، اثر امنیتی/مالی آن ارزیابی، فایل‌های Markdown و ADR مرتبط نسخه‌بندی، آزمون‌های لازم اجرا و در تاریخچهٔ ممیزی ثبت شود. سند قدیمی تا merge و انتشار اصلاحیهٔ مصوب، به‌صورت بی‌ردپا دور زده نمی‌شود.

### حدود اختیار و روش اجرای دستور
- مالک می‌تواند اولویت‌ها، دامنه، انتخاب معماری، providerهای مجاز، معیارهای پذیرش داخلی و Release Authorization را تصویب یا تغییر دهد؛ سیاست ریسک و سقف محافظتی فقط با تغییر صریح و نسخه‌بندی‌شدهٔ مورد تأیید مالک قابل تغییرند، نه با فرمان مدل در زمان اجرا.
- دستور مالک برای فعال‌سازی live به‌عنوان **درخواست/مجوز آغاز runbook انتشار** تلقی می‌شود. سامانه باید بدون بحث زائد مسیر عملیاتی مصوب را اجرا کند، اما فقط در صورت سبز بودن release gates، صحت اتصال/داده، تأیید risk supervisor، وجود credentialهای محلی معتبر و نبود مانع قانونی/قراردادی. اگر گیت اجباری شکست بخورد، اجرا باید متوقف و علت قابل‌اقدام گزارش شود؛ متن یا دستور محاوره‌ای به‌تنهایی گیت را خاموش نمی‌کند.
- ChatGPT دسترسی خودکار به سیستم محلی، secretها یا حساب صرافی ندارد و نمی‌تواند از داخل گفتگو معاملهٔ زنده را فعال کند. می‌تواند سند، تغییر پیشنهادی، checklist و runbook بسازد؛ اجرای واقعی در runtime تحت کنترل مالک و از مسیر release تأییدشده انجام می‌شود.
- این بند اولویت مالک را در **حاکمیت داخلی پروژه** تثبیت می‌کند، اما مقررات و قوانین قابل‌اعمال، شرایط صرافی/provider، امنیت اعتبارنامه‌ها و کنترل‌های فنی غیرقابل‌تخطی را حذف نمی‌کند. حذف یا تضعیف هر کنترل، خودش تغییر پرریسک و نیازمند بررسی/تصویب جداگانه است و نباید به‌عنوان اجرای فوری live تفسیر شود.

### ردپای ممیزی Owner Directive
هر دستور تغییردهندهٔ معماری باید شامل `directive_id`, `owner`, `timestamp`, `affected_docs/ADRs`, `risk_impact`, `security_impact`, `tests_required`, `approval_status`, `commit_sha` و نتیجهٔ GO/NO-GO باشد. برای درخواست live علاوه بر این موارد، release version، config/model hash، سقف محافظتی فعال، محیط، صرافی/حساب هدف و روش rollback باید ثبت شود.

# ۲. معماری کلان (Macro Architecture)
## ۲.۱ دیاگرام مفهومی
```text
                    CONTROL PLANE (چندکاربره)
       هویت/مجوزها | ثبت حساب‌ها | Secret Manager | سیاست ریسک
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
   Risk Supervisor / Portfolio         Model Registry / Release Gate
          │                                     │
          └──────────────┬──────────────────────┘
                         │ سیاست/نسخهٔ تأییدشده
                         ▼
    WORKER ISOLATION: یک نمونه به ازای (tenant, account, exchange, env)
 ┌─────────────────────────────────────────────────────────────────────┐
 │ Freqtrade Core (orders, DB, sync, backtesting, execution lifecycle) │
 │     │                       │                                       │
 │     ▼                       ▼                                       │
 │ DataProvider / OHLCV    AliXStrategy                               │
 │     │                       │                                       │
 │     ▼                       │                                       │
 │ Data Quality Gate ─────► FreqAI / IFreqaiModel                     │
 │                             │                                       │
 │                  Primary forecast + regime + uncertainty            │
 │                             │                                       │
 │                  Triple Barrier / Meta-Label                        │
 │                             │                                       │
 │ External Data Connectors ← Connectivity Manager (policy-routed)     │
 │   ├─ Market analytics: CoinGecko / CoinMarketCap / CoinGlass         │
 │   ├─ On-chain / whale events: approved APIs                          │
 │   └─ RSS / calendar / X / Telegram / Persian news → News Store        │
 │                                      │                               │
 │                               NLP Service (local)                    │
 │                                      │                               │
 │                 Point-in-time feature join + External Feature Store  │
 │                             ▼                                       │
 │                Decision Proposal: ENTRY/HOLD/EXIT                   │
 │                             │                                       │
 │                    Risk Gate / Approval                             │
 │                             │ approval only                         │
 │                             ▼                                       │
 │        Freqtrade Order Lifecycle → Exchange Adapter / CCXT          │
 └─────────────────────────────────────────────────────────────────────┘
                   │                             │
                   ▼                             ▼
        Append-only Decision Ledger       Spot Exchange API
                    │
                    ▼
        Reconciled Daily Snapshot (read-only)
                    │
                    ▼
    Reporting Engine → KPI aggregation → PNG chart renderer
                    │
                    ▼
       Notification Service / Bale Bot API
       daily schedule: 07:00 configured timezone

   Novaryn Nobitex Adapter ──► adapter compatibility boundary ──► Freqtrade
```

## ۲.۲ جداسازی مسئولیت‌ها
| لایه | مالک مسئولیت | مرز ممنوع |
|---|---|---|
| Freqtrade | چرخهٔ معامله، order state، همگام‌سازی، database معاملات، backtest/dry-run | AliX نباید این منطق را از نو پیاده کند. |
| AliXStrategy | تبدیل خروجی مدل به سیگنال و callbackهای Freqtrade؛ اجرای گیت ورود | دسترسی مستقیم به REST صرافی یا ثبت سفارش موازی. |
| FreqAI | train/fit/predict، pipelineهای داده، lifecycle مدل | دسترسی به secret یا امکان تغییر policy. |
| Data Quality Gate | اعتبارسنجی OHLCV، feature vector و زمان‌بندی | اصلاح بی‌ردپای دادهٔ ناقص یا پنهان کردن gap. |
| NLP Service | ingest/normalization/classification ویژگی خبر | تولید یا ارسال سفارش. |
| Risk Gate | تصمیم قطعی allow/reject/resize و بررسی محدودیت‌ها | تغییرپذیری از سوی مدل یا LLM. |
| Risk Supervisor | توقف بیرونی ورود، پایش loss/exposure، اجرای runbook | اعتماد صرف به همان worker که باید متوقف شود. |
| Ledger | ثبت append-only تصمیم، ارزیابی و ارتباط شناسه‌ها | جایگزینی database رسمی معاملات Freqtrade. |
| Reporting Engine | ساخت snapshot، KPI و نمودارهای روزانه و تحویل از طریق Bale adapter | ایجاد سفارش، تغییر policy یا اعلام PnL قطعی از دادهٔ reconcileنشده. |
| Notification Service / Bale adapter | تحویل امن پیام/تصویر، retries محدود و ثبت delivery state | پذیرش فرمان معاملاتی از چت یا نگهداری token در log/Git. |

## ۲.۳ معماری Multi-User / Multi-Account / Multi-Exchange
**الگوی پیشنهادی:** اجرای چندین worker ایزولهٔ Freqtrade؛ هر worker به‌طور معمول نمایندهٔ یک ترکیب `(tenant_id, account_id, exchange_id, environment, strategy_version)` است. Control Plane حساب‌ها و مجوزها را مدیریت می‌کند، اما نباید کلیدها یا state معامله را بین workerها به اشتراک بگذارد.

### Control Plane
- مدیریت tenant، مالکیت حساب، مجوزها و وضعیت مجاز بودن معامله.
- نگاشت امن `account_id` به secret reference؛ مقدار واقعی کلید در تنظیمات عمومی یا مدل ذخیره نمی‌شود.
- انتشار نسخهٔ تأییدشدهٔ strategy/model/config و ثبت rollout/rollback.
- سیاست‌های tenant، account و portfolio؛ محدودیت‌های portfolio می‌تواند سخت‌گیرانه‌تر از محدودیت حساب باشد، نه ضعیف‌تر.

### Worker Plane
- process/container مجزا برای هر حساب/صرافی/محیط، با کلیدهای مختص همان حساب.
- پایگاه داده، فایل‌های FreqAI، cache، log و workspace مجزا؛ از اشتراک writable `user_data` بین workerها خودداری شود.
- مدل immutable و تأییدشده می‌تواند به‌صورت artifact فقط‌خواندنی توزیع شود؛ state آموزش و فایل‌های runtime مشترک نباشند.
- کاربر نهایی به سیستم‌عامل، secret store یا فایل‌های کاربر دیگر دسترسی ندارد.

### Risk Plane
- **Risk Gate حساب:** اعتبارسنجی stake/exposure/تعداد موقعیت و stop پیشنهادیِ پویا در برابر سقف محافظتی مطلق، موجودی آزاد، قواعد بازار و ریسک باقی‌مانده؛ این‌ها را نباید با عدد ثابتِ ورودی مالک اشتباه گرفت.
- **Portfolio Supervisor:** تجمیع loss، exposure و وضعیت halt across accounts؛ در صورت رسیدن به حد پرتفوی، اجازهٔ ورود جدید همهٔ workerهای مشمول را لغو می‌کند.
- برای جلوگیری از fail-open، خطا یا عدم دسترس‌پذیری کنترل‌پلین/ریسک به توقف ورود جدید منجر می‌شود.

### چرا یک worker برای همهٔ حساب‌ها نه؟
یک نمونهٔ مشترک باعث اختلاط کلیدها، state، پایگاه داده، مدل‌های cacheشده و دامنهٔ خرابی می‌شود. با workerهای ایزوله، رشد سامانه عمدتاً به افزایش تعداد workerها و ظرفیت observability نیاز دارد، نه تغییر ساختاری در موتور معامله. برای کنترل هزینه، ظرفیت worker و queue NLP را جداگانه مقیاس دهید.

## ۲.۴ مرز آداپتور صرافی
تعریف داخلی AliX برای صرافی باید قرارداد مشترک داشته باشد: `market metadata`, `OHLCV`, `balances`, `open orders`, `submit`, `cancel`, `order status`, `fills`, `fees`, `precision/limits`, `rate limits`, `clock/time`, `idempotency/client order id` و `error taxonomy`.

این قرارداد **جایگزین CCXT نیست**؛ لایهٔ نرمال‌سازی و تست انطباق است. برای مسیر سفارش، تنها یک صاحب lifecycle وجود دارد: Freqtrade.

## ۲.۵ Data Quality Gate (لایهٔ مستقل)
Data Quality Gate باید هم پیش از ورود داده به feature engineering و هم درست پیش از inference فعال باشد. در مسیر آموزش، کنترل‌ها روی کل بازه اعمال می‌شود؛ در زمان اجرا، هر کندل/بردار ویژگی باید status و دلیل اعتبار داشته باشد.

### کنترل‌های الزامی بازار
- تازگی: timestamp آخرین کندل با timeframe و تأخیر قابل قبول منطبق باشد؛ cutoff برای هر timeframe به‌صورت config نسخه‌بندی شود.
- ترتیب زمانی: monotonic بودن timestamp، حذف duplicateها و مشخص بودن timezone؛ یک کندل بسته‌نشده با کندل بسته‌شده اشتباه نشود.
- Gap: شمارش کندل‌های گمشده و فاصله‌های غیرعادی؛ ابتدا تلاش برای دریافت مجدد از منبع، سپس رد window یا pair. **OHLCV گمشده به‌صورت پیش‌فرض با interpolation یا forward-fill ساخته نمی‌شود.**
- یکپارچگی OHLCV: مقادیر finite، حجم غیرمنفی، `high >= max(open, close)`, `low <= min(open, close)`, `high >= low` و قیمت‌های مثبت.
- نوسان/دادهٔ پرت: flag برای جهش نامحتمل و مقایسه با منبع/تایم‌فریم دیگر؛ حذف خودکار رویداد واقعی بازار مجاز نیست، تصمیم رد به policy تست‌شده وابسته است.
- schema: نام/نوع ستون، واحد اندازه‌گیری، ارز مظنه، precision، timeframe، exchange، pair و feature version باید تطبیق داشته باشد.
- پوشش پنجره: تعداد ردیف و نسبت مقادیر null/NaN/inf در پنجرهٔ مدل از حداقل قراردادی کمتر نباشد.

### کنترل‌های ویژهٔ NLP
- زمان `published_at`, `ingested_at`, `available_at` و `processed_at` جدا ثبت شوند.
- در decision time، فقط خبرهایی مجازند که `available_at <= decision_time` باشند.
- منبع، hash محتوا، language، asset mapping، event type، مدل/نسخه و confidence ثبت شود.
- خبر نامرتبط، تکراری، timestamp نامعتبر یا خروجی با confidence ناکافی باید حذف/neutral شود و به شکل مثبت پیش‌فرض تبدیل نشود.
- API خبری down یا NLP timeout شد: معامله بر اساس feature جعلی/قدیمی انجام نمی‌شود. Policy می‌تواند با ویژگی خبری missing مشخص‌شده ادامه دهد فقط اگر همان حالت در backtest و validation آزموده و صریحاً تصویب شده باشد؛ در غیر این صورت ورود متوقف است.

### خروجی Gate
یک status استاندارد مانند `VALID`, `DEGRADED_ALLOWED` یا `REJECTED` همراه با `reason_codes`, `checked_at`, `data_version` و `window_fingerprint` تولید شود. تنها `VALID` یا حالت degraded دقیقاً تأییدشده اجازهٔ ورود به مدل/سیاست را دارد. رأی gate در Ledger ذخیره می‌شود. Gate برای هر provider خارجی freshness، schema، واحد/ارز، pagination gaps، duplicate ID، source status و مجوز پردازش را نیز بررسی می‌کند.

## ۲.۶ Connectivity Manager (لایهٔ مدیریت اتصال)

### هدف
Connectivity Manager یک لایهٔ مستقل برای انتخاب، health-check و پایش مسیر شبکهٔ هر سرویس است؛ از جمله API صرافی، بازارهای عمومی، منبع خبر، سرویس NLP و Model Registry. هدف، **تاب‌آوری و مشاهده‌پذیری اتصال** است؛ نه دور زدن محدودیت‌های قراردادی، جغرافیایی، تجاری، احراز هویت یا rate limit.

### روش‌های اتصال پشتیبانی‌شده
- **Direct:** مسیر مستقیم با TLS معتبر و DNS/egress قابل پایش.
- **HTTP/HTTPS Proxy:** پراکسی سازمانی یا متعلق به مالک با احراز هویت و پیکربندی مجزا برای هر مقصد.
- **SOCKS Proxy:** شامل SOCKS5 در صورت پشتیبانی کتابخانهٔ شبکه و مسیر REST/WebSocket موردنیاز.
- **VPN:** تونل شبکهٔ تعریف‌شده توسط عملیات، با route policy و خروجی پایدار؛ تنظیمات حساس و کلیدها خارج از Git نگهداری شوند.
- **SSH Tunnel:** port-forwarding برای endpointهای مشخص و مجاز؛ تنها به‌عنوان transport کنترل‌شده، نه جایگزین TLS یا سازوکار امنیت برنامه.

پشتیبانی معماری به معنی پشتیبانی خودکار همهٔ روش‌ها در هر نسخهٔ Freqtrade/CCXT نیست. سازگاری هر transport برای کتابخانه و endpoint واقعی باید در ماتریس آزمون ثبت شود. مرجع proxy در CCXT توضیح می‌دهد که HTTP/SOCKS قابل پیکربندی است و intermediary می‌تواند latency اضافه کند. [R23]

### سیاست مسیریابی و failover
1. برای هر `service_id` و `environment` یک route policy نسخه‌بندی‌شده تعریف می‌شود؛ مسیرهای مجاز، timeout، health probe، بودجهٔ retry، DNS و egress IP مورد انتظار مشخص هستند.
2. failover فقط میان مسیرهای از پیش تأییدشده و از نظر قرارداد/امنیت مجاز است. تغییر مسیر نباید باعث عبور از rate limit، محدودیت اشتراک، allowlist یا policy provider شود.
3. اگر درخواست سفارش timeout شود یا نتیجهٔ ارسال نامعلوم بماند، **تغییر route و ارسال مجدد سفارش ممنوع است تا reconciliation وضعیت سفارش از مسیر مورد تأیید انجام شود**. این قاعده از سفارش تکراری جلوگیری می‌کند.
4. برای API keyهای IP-restricted، egress IP ثابت/شناخته‌شده باید از پیش در صرافی مجاز شده باشد. پراکسی عمومی و ناشناس برای endpoint معاملاتی ممنوع است.
5. TLS certificate validation غیرفعال نمی‌شود؛ secrets پراکسی/VPN/SSH در Secret Manager یا secret mount هستند. هر route change، خطای احراز هویت، egress IP و latency در لاگ ساختاریافته ثبت می‌شود؛ credential و payload حساس log نمی‌شوند.
6. منبع جایگزین تنها در صورت یکسان بودن معنا/واحد/تازگی و گذشتن از Data Quality Gate قابل استفاده است. failover شبکه، مجوز استفاده از منبع دادهٔ جایگزین را ایجاد نمی‌کند.

### ساختار مسئولیت
- `connectivity/manager/`: route policy، health state، failover controller، circuit breaker و telemetry.
- `connectivity/proxy/`: adapterهای HTTP/HTTPS/SOCKS و اعتبارسنجی پیکربندی.
- `connectivity/vpn/`: template و runbook برای VPN؛ بدون private key در repository.
- `connectivity/tunnel/`: SSH tunnel profiles/healthcheck؛ کلیدها فقط از secret mount خوانده می‌شوند.

### معیار پذیرش اتصال
آزمون Direct، HTTP Proxy، SOCKS Proxy، VPN و SSH Tunnel باید برای endpointهای موردنیاز انجام شود: DNS/TLS، authentication، REST، WebSocket در صورت استفاده، latency/jitter، rate-limit handling، egress IP، reconnect، route health و رفتار هنگام شکست. مسیر نامعتبر باید fail-closed شود. در فازهای ۰ و ۱ این لایه به ترتیب طراحی/policy و پیاده‌سازی/آزمون پایه را می‌گذراند؛ پیش از آن اتصال معاملهٔ زنده مجاز نیست.

## ۲.۷ External Data & Social Media (لایهٔ داده‌های خارجی)

### کاتالوگ منابع پیشنهادی
منابع زیر **کاندیدای اتصال** هستند، نه منابعی که همگی از ابتدا فعال یا رایگان فرض شوند. هر مورد باید از نظر مجوز استفاده، پلن API، نرخ درخواست، تأخیر، retention، بازتوزیع، کیفیت و پوشش بازار ثبت و تأیید شود.

| خانواده | منابع | نحوهٔ استفادهٔ پیشنهادی |
|---|---|---|
| داده و تحلیل بازار | TradingView، CoinGecko، CoinMarketCap، CoinGlass | شاخص‌های کمکی بازار/مشتقات/نوسان و تحلیل بین‌منبعی؛ قیمت و وضعیت اجرایی نهایی همچنان از صرافی معامله‌گر/Freqtrade گرفته شود. |
| داده‌های آنچین و نهنگ‌ها | On-Chain Data از provider مجاز، Whale Alert | جریان رویدادهای زنجیره، انتقال‌های بزرگ و ویژگی‌های زمان‌مند؛ ثبت chain، asset, event_time و provider provenance. |
| اخبار و تقویم | RSS News، Economic Calendar، APIهای مجاز خبری | استخراج رویداد، headline/metadata، رویدادهای کلان اقتصادی و تحلیل متن در صورت مجوز. |
| شبکه‌های اجتماعی | Twitter/X API، Telegram Bot/API یا connector مجاز، Social Media APIs | سیگنال‌های sentiment/trend با احتساب نویز، bot/spam، manipulation و شرایط پلتفرم. دسترسی به کانال/داده باید مجاز باشد. |
| منابع ایرانی | سایت‌های تحلیل و خبر فارسی، RSS یا API رسمی آن‌ها | متن فارسی و خبرهای محلی؛ نگهداری snapshot و استفادهٔ مدل تنها در چارچوب شرایط منبع و حقوق نشر/پردازش. |

### قید حیاتی TradingView
TradingView در catalog ثبت می‌شود، اما طبق شرایط منتشرشدهٔ فعلی، استفادهٔ non-display از محتوا/داده برای تصمیم الگوریتمی یا معاملهٔ خودکار محدود شده است. بنابراین تا زمان وجود مجوز صریح و قابل استناد که use case پردازش ماشینی/تصمیم معاملاتی AliX را مجاز کند، دادهٔ TradingView **به‌عنوان ورودی خودکار مدل، Risk Gate، سفارش یا ارزیابی قیمت استفاده نمی‌شود**. APIهای community یا scraping endpointهای غیررسمی جایگزین مجوز رسمی نیستند. [R15]

### مسیر ورود داده و جداسازی stores
- داده‌های عددی بازار/تحلیل به `External Feature Store` یا analytics store می‌روند و با timestamp/source/version به feature contract متصل می‌شوند.
- خبر، RSS، تقویم اقتصادی و متن شبکه‌های اجتماعیِ مجاز ابتدا به `News Store` می‌روند؛ سپس NLP Service آن‌ها را به sentiment/event features نسخه‌بندی‌شده تبدیل می‌کند.
- همهٔ connectorها از Connectivity Manager و Data Quality Gate عبور می‌کنند. هیچ feed خارجی اجازهٔ دور زدن gate یا تغییر مستقیم order state ندارد.
- هر رکورد باید حداقل `source_id`, `provider`, `license_status`, `event_id`, `published_at`, `ingested_at`, `available_at`, `processed_at`, `asset_mapping`, `content_hash`, `schema_version` و status کیفیت داشته باشد. برای دادهٔ عددی، واحد، currency/quote, interval, market, event time, ingestion time و revision نیز ثبت می‌شود.
- برای کاهش latency از push/webhook/stream فقط در صورت پشتیبانی رسمی، cache، batching و inference غیرهم‌زمان استفاده شود؛ SLA بر پایهٔ اندازه‌گیری تعیین می‌شود، نه ادعای «پیش‌بینی فوری». خبر یا دادهٔ دیررس فقط از اولین زمان واقعی در دسترس بودن قابل استفاده است.
- در برابر rate limit از quota، cache، retry با backoff و circuit breaker استفاده می‌شود؛ جابه‌جایی حساب/پراکسی برای فرار از محدودیت مجاز نیست. از scraping، session cookie شخصی یا جمع‌آوری دادهٔ خصوصی/محدودشده بدون مجوز استفاده نشود.
- providerهای بیرونی منبع حقیقت قیمت fill یا موجودی نیستند؛ reconciliation سفارش و دارایی همیشه با API صرافی معامله‌گر انجام می‌شود.

### انتخاب مرحله‌ای منابع
نسخهٔ اول با **حداقل یک منبع بازار بیرونی و یک feed خبری مجاز** شروع می‌شود. منابع جدید به‌صورت adapter مستقل اضافه می‌شوند؛ هر connector دارای contract tests، freshness/SLO، license record، rate-limit policy، health metrics، historical snapshot strategy و fallback state است. اثر افزودهٔ آن در ablation/backtest اندازه‌گیری می‌شود؛ نبود بهبود یا هزینه/ریسک حقوقی نامتناسب دلیل غیرفعال‌سازی است.

## ۲.۸ Reporting Engine / Notification Service (گزارش‌دهی و اعلان‌ها)

Reporting Engine از Freqtrade DB، Decision Ledger، Risk Supervisor و snapshot موجودی/سفارش پس از reconciliation یک گزارش وضعیت روزانه می‌سازد. این سرویس فقط خوانندهٔ داده است؛ حق ارسال سفارش، تغییر ریسک، خاموش‌کردن هشدار یا کنترل live mode از طریق پیام‌رسان ندارد.

### زمان‌بندی و ارسال به بله
- گزارش هر روز در **ساعت ۰۷:۰۰** به وقت `reporting.timezone` تولید می‌شود. منطقهٔ زمانی با نام IANA در تنظیمات نسخه‌بندی می‌شود؛ مقدار اولیهٔ پیشنهادی `Asia/Tehran` است و مالک در فاز صفر آن را تأیید/اصلاح می‌کند. استفاده از UTC offset ثابت برای زمان‌بندی روزانه توصیه نمی‌شود.
- تحویل از طریق adapter اختصاصی `integrations/notifications/bale/` و Bale Bot API انجام می‌شود. در فاز پیاده‌سازی، مستندات رسمی جاری، قابلیت ارسال عکس/فایل، محدودیت حجم، نرخ مجاز و شرایط استفاده دوباره بررسی می‌شود. هر SDK ثالث فقط پس از بررسی مجوز، وابستگی‌ها و آزمون قرارداد انتخاب می‌شود؛ هیچ بستهٔ خاصی از پیش تأییدشده نیست.
- گزارش از یک متن خلاصه به همراه نمودارهای PNG و در صورت لزوم فایل/ضمیمهٔ جزئیات تشکیل می‌شود. اگر روش ارسال تصویر در API/SDK منتخب در دسترس نبود، fallback به ارسال فایل/گزارش متنی طبق قرارداد تست‌شده است؛ شکست ارسال باید ثبت و retry محدود انجام شود.
- گیرنده با `Bale chat_id` از قبل مجازشده مشخص می‌شود. Bot token و chat ID حساس در Secret Manager / local secret mount قرار می‌گیرند و هرگز در Git، log، screenshot یا خود گزارش درج نمی‌شوند. در نسخهٔ اول، بله **کانال اعلان یک‌طرفه** است و پیام دریافتی مجوز معامله یا تغییر پیکربندی نیست.

### محتوای گزارش گرافیکی
1. خلاصهٔ مدیریتی: وضعیت `RUNNING / DEGRADED / HALTED`, timestamp snapshot، صرافی/حساب با alias غیرحساس و وضعیت همگام‌سازی.
2. سود و زیان: realized PnL، unrealized PnL، هزینه/کارمزد و net PnL روز، نمودار equity و PnL دوره‌ای؛ مقادیر فقط پس از reconciliation با دادهٔ معتبر محاسبه می‌شوند.
3. معاملات: تعداد ورود/خروج و معاملات بسته‌شده، برد/باخت، میانگین سود/زیان و وضعیت سفارش‌های باز/ناموفق؛ معیارهایی مانند win-rate همراه با نمونهٔ معاملات نمایش داده می‌شوند تا برداشت آماری گمراه‌کننده ایجاد نشود.
4. سرمایه و ریسک: equity/balance تجمیعی، exposure، drawdown جاری، مصرف سقف محافظتی مطلق، وضعیت Kill Switch و تعداد ردهای Risk Gate. credential یا شناسهٔ حساس سفارش به گزارش ارسال نمی‌شود.
5. سلامت فنی و مدل: نسخهٔ strategy/model، آخرین heartbeat، سلامت Data Quality Gate، freshness/gaps، latency/error rates، وضعیت Connectivity Manager و تأخیر News/NLP در صورت فعال بودن.

### کیفیت، قابلیت اطمینان و ممیزی
- هر گزارش با `report_id`, `report_date`, `account_scope`, `generated_at`, `data_cutoff`, `source_snapshot_hash`, `model/config revision` و `delivery_status` ثبت می‌شود؛ رکورد شامل token یا اطلاعات احراز هویت نیست.
- اگر reconciliation ناموفق، منبع داده stale یا دفتر معاملات ناسازگار باشد، گزارش باید برچسب واضح `DATA NOT RECONCILED / DEGRADED` داشته باشد؛ ارقام نامطمئن نباید به‌عنوان PnL قطعی ارائه شوند.
- retry با backoff و سقف تعداد تلاش‌ها انجام می‌شود؛ کلید idempotency منطقی بر مبنای تاریخ/حساب/نسخه از ارسال چندبارهٔ ناخواسته جلوگیری می‌کند. شکست نهایی در local log/monitoring علامت می‌خورد و در صورت وجود کانال دوم تأییدشده می‌تواند اعلان دهد.
- نمودار و snapshot باید از دادهٔ همان زمان تولید شوند؛ اگر فرایند پس از ۰۷:۰۰ دوباره اجرا شد، گزارش اصلاحی با revision متفاوت ارسال می‌شود و گزارش قبلی بازنویسی بی‌ردپا نمی‌شود.
- گزارش روزانه یک ابزار مشاهده‌پذیری است، نه جایگزین پایش لحظه‌ای، alert بحرانی یا Risk Supervisor.

### معیار پذیرش
- scheduler در روزهای عادی و تغییر ساعت تابستانی/منطقه‌ای، در ساعت محلی پیکربندی‌شده اجرا شود؛ timezone policy و رفتار restart ثبت شده باشد.
- ارسال پیام متنی، PNG و fallback ضمیمه از mock/contract tests بگذرد؛ خطاها token/chat ID را در log افشا نکنند.
- جمع PnL و تعداد معاملات با source of truth و snapshot تعریف‌شده تطبیق داشته باشد؛ mismatch به هشدار صریح منجر شود.
- گزارش فقط به chat ID allowlist شده ارسال شود؛ آزمون منفی گیرندهٔ غیرمجاز موفق باشد.
- ارسال تکراری، Bale API down، token نامعتبر، file size rejection، timeout و restart scheduler آزمون شده باشند.

# ۳. ساختار پوشه‌ها (Directory Structure)
ساختار زیر، پوشه‌های استاندارد Freqtrade را حفظ می‌کند. کد آداپتور Novaryn یا باید به‌صورت submodule وابسته به commit مشخص نگه داشته شود یا پس از بررسی مالکیت/مجوز در پکیج داخلی نسخه‌بندی‌شده قرار گیرد؛ URL مخزن هنوز در سند حاضر ثبت نشده است.

```text
AliX/
├── .github/{workflows/, ISSUE_TEMPLATE/, PULL_REQUEST_TEMPLATE.md}
├── docs/{MRD.md, adr/, runbooks/, threat-model/}
│   ├── data-provider-registry.md
│   ├── connectivity-test-matrix.md
│   └── phase-reports/
├── control_plane/                 # tenants, accounts, secret refs, rollout
├── risk/                          # immutable policy, supervisor, audit
├── connectivity/
│   ├── manager/                    # route policy, health, failover, telemetry
│   ├── proxy/                      # HTTP/HTTPS/SOCKS proxy adapters
│   ├── vpn/                        # VPN templates/runbooks; no private keys
│   └── tunnel/                     # SSH tunnels/port-forwarding; secret mounts
├── external_data/
│   ├── analytics/                  # CoinGecko/CMC/market analytics adapters
│   ├── market_intelligence/        # on-chain, whale, derivatives, calendar
│   ├── news/                       # RSS and licensed news APIs
│   └── sentiment/                  # X/Telegram/social/Persian-language feeds
├── services/
│   ├── data_quality/ | news_ingestion/ | nlp_inference/
│   ├── model_registry/             # immutable artifacts and promotion
│   └── reporting/
│       ├── scheduler/               # daily 07:00, IANA timezone
│       ├── snapshots/               # reconciled metrics; no credentials
│       ├── charts/                  # PNG/equity/PnL/operations charts
│       └── templates/               # versioned text and chart layouts
├── integrations/{exchanges/, nobitex/}  # contracts + adapter tests
├── integrations/notifications/bale/ # Bale Bot API client boundary
├── vendor/novaryn-nobitex/         # pinned submodule; reuse existing adapter
├── deployment/{compose/, policies/, runbooks/}
├── config/{common/, environments/, markets/, tenant-templates/}
├── user_data/                      # separate writable volume per worker
│   ├── strategies/AliXStrategy.py
│   ├── freqaimodels/{AliXCompositeModel.py, AliXTripleBarrierModel.py}
│   └── data/, models/, backtest_results/, logs/, trades.sqlite
├── contracts/                      # feature/model/order schemas
├── tests/{unit,integration,contracts,risk,recovery,backtesting}/
├── artifacts-manifest/             # manifests/hashes; no large weights in Git
├── pyproject.toml / requirements.lock
└── README.md
```

## قواعد placement
- FreqAI modelها: `user_data/freqaimodels/`; strategy: `user_data/strategies/`.
- transport وroute policy در `connectivity/` قرار می‌گیرند؛ فایل‌های واقعی VPN، private key، password، token و proxy credential هرگز در Git نیستند. فقط template بدون secret و runbook نسخه‌بندی می‌شوند.
- provider adapterهای بازار در `external_data/analytics/`، منابع آنچین/مشتقات/تقویم در `market_intelligence/`، خبر در `news/` و sentiment در `sentiment/` قرار می‌گیرند. تمام مسیرها از یک client مشترک با rate limit، retries، tracing و contract tests استفاده می‌کنند.
- `docs/data-provider-registry.md` باید منبع، endpoint رسمی، وضعیت مجوز (`approved/conditional/blocked`)، پلن، نرخ مجاز، retention، تاریخ بررسی و مالک integration را ثبت کند. TradingView تا وقتی مجوز non-display صریح ندارد با وضعیت `blocked_for_ai_trading` می‌ماند.
- مدل NLP را ترجیحاً از `user_data` جدا و به‌عنوان artifact readonly در مسیر سرویس NLP mount کنید؛ اگر نیاز به نگهداری manifest داخل `user_data/nlp/` است، وزن‌های حجیم را در Git commit نکنید.
- هیچ secret در `config/*.json`, Docker image, Git, notebook یا Decision Ledger ذخیره نشود.
- `vendor/novaryn-nobitex` باید با commit SHA ثابت باشد؛ branch متحرک یا install بدون pin در release استفاده نشود.
- مسیر state هر worker شامل tenant/account/exchange/environment باشد و permission filesystem حداقلی تنظیم گردد.
- `services/reporting/` فقط از snapshotهای reconcileشده/نسخه‌بندی‌شده می‌خواند؛ خروجی PNG و metadata گزارش artifact هستند اما bot token، chat ID خصوصی و balance جزئی حساس وارد Git نمی‌شوند.
- `integrations/notifications/bale/` صرفاً transport یک‌طرفه برای ارسال گزارش/alert است؛ هیچ handler پیام دریافتی مجاز به تغییر live mode، risk policy یا ثبت سفارش نیست.
- scheduler گزارش روزانه timezone نام‌دار IANA دارد؛ در Phase 0 `reporting.timezone` و chat allowlist تعیین و در secret/config امن نگهداری می‌شود.

# ۴. لایهٔ هوش مصنوعی (AI Core & Intelligence Layer)
## ۴.۱ لایه‌های تصمیم AI
1. **Market State / Regime:** تخمین شرایط بازار از ویژگی‌های گذشته‌نگر مانند volatility، trend، liquidity proxy و ساختار کندل؛ خروجی کلاس/احتمال رژیم است، نه دستور سفارش.
2. **Primary Forecast:** برآورد بازده، جهت یا احتمال حرکت در افق تعیین‌شده. target، افق و unit باید نسخه‌بندی شوند.
3. **Opportunity / Meta-Label:** تخمین احتمال قابل معامله بودن proposal مدل اصلی با لحاظ هزینه، توقف، horizon و کیفیت داده.
4. **News Context:** امتیاز احساسات، کلاس رویداد، دارایی مرتبط و confidence از NLP، با زمان دسترس‌پذیری دقیق.
5. **Decision Policy:** ترکیب خروجی مدل‌ها، عدم‌قطعیت، رژیم، خبر و قواعد حالت. خروجی فقط پیشنهاد ساختاریافته است: `action`, `pair`, `confidence`, `expected_edge`, `suggested_stake_fraction`, `model_ids`, `feature_timestamp`.
6. **Risk Gate:** مجوز قطعی و مستقل. حتی اگر خروجی AI confidence کامل داشته باشد، سقف‌ها و وضعیت توقف بر آن مقدم‌اند.

## ۴.۲ جریان از خبر تا تصمیم
منابع ورودی در connector registry شامل RSS News، تقویم اقتصادی، APIهای خبری، X/Twitter API، Telegram connector مجاز و وب‌سایت‌های تحلیلی/خبری فارسی با API یا RSS مورد تأیید هستند. داده‌های market analytics از CoinGecko، CoinMarketCap و CoinGlass می‌توانند از مسیر جداگانه به External Feature Store متصل شوند. TradingView تا زمان تأیید کتبی مجوز استفادهٔ ماشینی، از ingestion معاملاتی غیرفعال می‌ماند. [R15-R22]

1. News provider خبر خام را با URL/شناسه، زمان منبع و hash وارد می‌کند.
2. Ingestion timestamp دریافت محلی را ثبت و duplicateها را با ID/hash حذف می‌کند؛ نسخهٔ خام یا snapshot مجاز برای بازتولید نگه داشته می‌شود.
3. Resolver خبر را به نماد/دارایی‌ها، زبان، نوع رویداد و منابع مرتبط نگاشت می‌کند. ابهام mapping به `unknown/unmapped` منجر می‌شود، نه برچسب BTC پیش‌فرض.
4. NLP محلی روی متن نرمال‌شده inference انجام می‌دهد. خروجی شامل `sentiment_label`, `sentiment_score`, `confidence`, `event_type`, `model_id`, `model_revision`, `processed_at` و `content_hash` است.
5. Point-in-time join بر اساس `available_at` داده را فقط به پنجرهٔ زمانی مجاز متصل می‌کند. در backtest، snapshot تاریخی همان زمان لازم است؛ دریافت امروز خبر قدیمی دلیل کافی برای استفاده از آن در backtest گذشته نیست.
6. Data Quality Gate زمان‌سنجی، duplicate، schema و تازگی خبر را کنترل می‌کند.
7. ویژگی‌های خبری به قرارداد feature با feature-set version افزوده می‌شوند. دسترسی مدل بازار به خروجی NLP فقط از طریق این قرارداد است، نه از طریق HTTP خام یا متن دلخواه.
8. FreqAI مدل را اجرا و خروجی همراه uncertainty/probability تولید می‌کند. Decision Policy آن را به proposal تبدیل می‌کند.
9. Risk Gate proposal را با حدود account/portfolio و حالت worker تطبیق می‌دهد و رأی را در Ledger ذخیره می‌کند.
10. فقط پس از تأیید Risk Gate، مسیر استاندارد strategy و Freqtrade اجازهٔ ورود می‌دهد؛ Freqtrade مدیریت order/fill/cancel و reconciliation را انجام می‌دهد.

## ۴.۳ قرارداد دادهٔ خبری
| فیلد | معنا / الزام |
|---|---|
| `event_id` | شناسهٔ پایدار خبر/رویداد |
| `source_id`, `source_url` | منبع مجاز و امکان audit؛ URL ممکن است براساس retention policy حذف/ماسک شود. |
| `published_at` | زمان ادعاشده توسط ناشر |
| `ingested_at` | زمان دریافت در سامانه |
| `available_at` | اولین زمانی که این متن واقعاً برای تصمیم قابل دسترس بود؛ مبنای جلوگیری از look-ahead |
| `processed_at` | زمان خاتمهٔ NLP |
| `asset_mapping` | فهرست دارایی‌های نگاشت‌شده با confidence |
| `sentiment_score`, `sentiment_label` | ویژگی مدل، نه حقیقت قطعی بازار |
| `event_type`, `confidence` | نوع رویداد و confidence مدل |
| `model_id`, `model_revision` | شناسه و commit دقیق وزن/tokenizer/config |
| `content_hash`, `schema_version` | dedupe، integrity و سازگاری |

## ۴.۴ استفاده از FreqAI
- مدل سفارشی باید از interface نسخهٔ قفل‌شدهٔ `IFreqaiModel` مشتق شود و `train`, `fit`, `predict` و چرخهٔ دادهٔ نسخهٔ مورد استفاده را رعایت کند. امضای دقیق متدها قبل از پیاده‌سازی با commit هدف تأیید می‌شود؛ به API حافظه‌محور یا نسخهٔ قدیمی تکیه نشود. [R1]
- از `feature_engineering_expand_all`, `feature_engineering_expand_basic`, `feature_engineering_standard` و `set_freqai_targets` مطابق مستندات نسخهٔ انتخاب‌شده استفاده می‌شود. Featureها باید گذشته‌نگر بمانند و targetها از featureها جدا باشند. [R2]
- مدل، scaler/pipeline، feature list و target schema باید در کنار هم نسخه‌بندی شوند. پیش‌بینی با schema یا مدل نامنطبق رد می‌شود.

## ۴.۵ DeepAlpha: پذیرش مشروط
DeepAlpha نامزد آماده‌سازی Triple-Barrier، Meta-Labeling، انتخاب ویژگی و اعتبارسنجی زمانی است؛ اما پیش از استفاده، باید بستهٔ دقیق، commit، license، نحوهٔ نصب، شیوهٔ ثبت مدل، dependency tree و سازگاری با نسخهٔ Freqtrade قفل‌شده ممیزی شود. صفحهٔ PyPI و راهنمای مخزن عمومی، در مسیر استقرار مدل رفتار یکسانی را توصیف نمی‌کنند؛ بنابراین تصمیم بر پایهٔ یک نام بسته یا ادعای README کافی نیست. [R6][R7]

**گیت پذیرش DeepAlpha:** (الف) نصب تمیز در virtualenv/container بازتولیدپذیر؛ (ب) بدون patch فایل‌های داخلی Freqtrade؛ (پ) train/predict/backtest و reload؛ (ت) آزمون feature/target و no-look-ahead؛ (ث) save/load مدل و سازگاری با worker ایزوله؛ (ج) مجوز وابستگی‌ها؛ (چ) عملکرد بیرون از نمونه در برابر baseline.

## ۴.۶ Plan B: Triple-Barrier و Meta-Labeling با امکانات FreqAI
اگر DeepAlpha با نسخهٔ هدف سازگار نبود یا ریسک نگهداری/مجوز آن پذیرفته نشد، پروژه متوقف نمی‌شود و به آن وابسته نمی‌ماند. مسیر Plan B یک مدل ترکیبی سفارشی FreqAI است:

- **Triple-Barrier target:** در data/target layer، برچسب هر فرصت بر اساس برخورد به حد بالایی سود، حد پایینی ضرر یا پایان افق زمانی تولید شود. barrierها می‌توانند با volatility هدف‌گذاری شوند، اما پارامترها فقط با دادهٔ گذشته و در train/validation محاسبه می‌شوند. منطق برچسب‌گذاری به دادهٔ آینده برای ساخت `y` دسترسی دارد؛ این آینده هرگز وارد feature ورودی مدل نمی‌شود.
- **Primary model:** پیش‌بینی فرصت/جهت یا score اولیه را تولید می‌کند. سیگنال in-sample برای آموزش Meta-Model مجاز نیست.
- **Out-of-fold signal generation:** در هر fold زمانی، مدل primary تنها روی بخش گذشته آموزش می‌بیند و روی بخش بعدی پیش‌بینی می‌کند. این پیش‌بینی OOF به‌عنوان ورودی آموزشی Meta-Label استفاده می‌شود.
- **Meta target:** پاسخ دودویی/چندکلاسه به این پرسش که «اگر proposal اولیه در آن لحظه اجرا می‌شد، آیا با احتساب fee/slippage و قرارداد خروج/Triple-Barrier، نتیجهٔ قابل قبول داشت؟»
- **Composite `IFreqaiModel`:** یک کلاس سفارشی چرخهٔ `fit/predict` را برای دو مرحله هماهنگ می‌کند یا دو artifact نسخه‌بندی‌شده را مدیریت می‌کند. همهٔ artifactها باید از مسیر lifecycle و ذخیره‌سازی نسخهٔ منتخب FreqAI قابل reload باشند. پشتیبانی دقیق از چند artifact و schema باید در PoC اثبات شود؛ monkey patch هسته ممنوع است.
- **Purging/embargo و walk-forward:** نمونه‌های train که بازهٔ target آنها به validation overlap می‌کند purge می‌شوند؛ embargo زمانی بین بخش‌ها اعمال می‌شود. معیار پذیرش، بهبود قابل تکرار در چند پنجرهٔ زمانی است، نه یک backtest انتخابی.
- **حالت توقف:** تا پایان PoC، اگر مدل MetaLabel یا OOF artifact در دسترس/معتبر نباشد، هیچ ورود جدیدی مجاز نیست.

اگر پیاده‌سازی Composite Model با interface نسخهٔ قفل‌شده سازگار نبود، Plan B به دو سرویس/پکیج مستقل در pipeline research تقسیم می‌شود و فقط خروجی artifactهای immutable و از پیش اعتبارسنجی‌شده را به worker زنده می‌دهد؛ به شرط آن‌که interface زنده، latency و point-in-time semantics آزموده شده باشد. بازنویسی FreqAI راهکار جایگزین نیست.

## ۴.۷ مدل‌های NLP با مجوز مناسب برای بررسی تجاری
مدل‌های زیر **نامزدهای فنی** هستند و هیچ‌کدام صرفاً با دیدن badge مجوز برای production تأیید نمی‌شوند. مجوز وزن، مجوز dataset، شرایط منبع خبر و تعهدات attribution باید جدا بررسی شوند.

| نامزد | مجوز اعلام‌شده برای وزن/مخزن | کاربرد و محدودیت |
|---|---|---|
| `Kenpache/multilingual-financial-sentiment-v2` [R10] | Apache-2.0 در صفحهٔ مدل | مدل مالی چندزبانه و شامل زبان آلمانی/انگلیسی؛ مدل نسبتاً بزرگ‌تر، benchmark و provenance داده باید بررسی شود. |
| `AnkitAI/distilbert-base-uncased-financial-news-sentiment-analysis` [R9] | Apache-2.0 در صفحهٔ وزن | سبک‌تر و مناسب متن انگلیسی خبر مالی؛ model card دیتاست Financial PhraseBank را ذکر می‌کند و صریحاً بررسی مجوز dataset برای استفادهٔ تجاری را خواستار است. تا روشن شدن زنجیرهٔ داده، مجوز عملیاتی آن تأییدشده نیست. |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` [R11] | CC-BY-4.0 در صفحهٔ مدل | نامزد مکمل برای متن کوتاه/شبکهٔ اجتماعی و دارای attribution؛ مدل عمومی sentiment است، نه sentiment تخصصی خبر مالی. فقط بعد از benchmark دامنهٔ داده به کار رود. |

**ممنوعیت:** استفاده از `ProsusAI/finbert` به‌عنوان پیش‌فرض تجاری بدون مجوز اضافی، چون صفحهٔ مدل مجوز CC-BY-NC-SA-4.0 را اعلام می‌کند. [R12]

**فرایند انتخاب مدل:** ثبت model ID و immutable revision، hash وزن/tokenizer، license manifest، dataset lineage تا حد موجود، تست امنیت artifact، macro-F1 و calibration روی مجموعهٔ خبری خودمان، سنجش per-language/per-event، latency و memory، و بررسی حقوقی پیش از promotion.

# ۵. لایهٔ امنیت و مدیریت ریسک (Risk Engine & Safeguards)
## ۵.۱ اصل طراحی
مدل فقط proposal می‌دهد. سیاست ریسک با پیکربندی immutable و منطق قطعی کنترل می‌شود، دسترسی نوشتن آن به مدل داده نمی‌شود و در مسیر زنده به‌صورت gate مستقل اجرا می‌شود. **Fail-closed برای ورود جدید** اصل پیش‌فرض است.

## ۵.۲ Dynamic Risk Allocation Engine

### هدف و مرز اختیار
این موتور در لایهٔ هوش AliX، پیشنهاد پویا برای بودجهٔ ریسک روزانه، اندازهٔ موقعیت و حد ضرر تولید می‌کند. خروجی آن **پیشنهاد مدل** است، نه مجوز سفارش و نه ویرایش سیاست محافظتی. FreqAI، مدل تشخیص رژیم و Meta-Labeling می‌توانند دربارهٔ ریسک‌پذیری هر فرصت نظر بدهند؛ هیچ‌یک حق تغییر سقف مطلق سرمایه، خاموش‌کردن Kill Switch یا عبور مستقیم به API صرافی را ندارند.

### ورودی‌های تصمیم پویا
- احتمال/اطمینان تشخیص رژیم بازار و شاخص عدم‌قطعیت یا entropy مدل؛ confidence خام تا زمان کالیبراسیون OOS قابل استفاده به‌عنوان احتمال واقعی نیست.
- ATR و ATR نرمال‌شده بر قیمت، realized volatility، تغییرات volatility، gap و کیفیت OHLCV.
- edge پیش‌بینی‌شده پس از کارمزد، spread و slippage تخمینی؛ Meta-Label احتمال معتبر بودن سیگنال اولیه را به‌صورت جداگانه ارزیابی می‌کند.
- equity و نقدینگی واقعاً در دسترس، زیان تحقق‌یافته/تحقق‌نیافته، exposure و همبستگی پوزیشن‌های موجود، ظرفیت باقی‌مانده تا سقف محافظتی مطلق.
- نقدشوندگی، عمق/حجم قابل‌اتکا، محدودیت precision/minimum order صرافی و کیفیت/تازگی داده.

### تصمیم‌های پویا
1. **بودجهٔ ریسک روزانه:** در آغاز و طی روز به‌صورت پویا محاسبه می‌شود و با تغییر رژیم، نوسان، drawdown جاری، کیفیت فرصت و عدم‌قطعیت بازتنظیم می‌شود. این مقدار یک تخصیص داخلی مدل است، نه سقف روزانهٔ ثابتِ واردشده توسط مالک؛ حق مصرف بودجهٔ فراتر از ظرفیت باقی‌مانده تا سقف مطلق را ایجاد نمی‌کند.
2. **حد ضرر:** مدل/سیاست یادگرفته‌شده با توجه به رژیم، ATR و توزیع حرکت نامطلوب، حد ضرر پیشنهادی یا ضریب ATR را تعیین می‌کند. در صورت دادهٔ نامعتبر، stop نامعتبر یا فاصله‌ای که محاسبهٔ زیان آن از ظرفیت مجاز خارج شود، معامله رد یا اندازهٔ آن کاهش داده می‌شود.
3. **اندازهٔ موقعیت:** اندازه از بودجهٔ ریسک پیشنهادی و زیان تخمینی در فاصلهٔ stop به‌همراه کارمزد و buffer لغزش محاسبه می‌شود. برای Spot خطی، رابطهٔ مفهومی تقریباً چنین است: `stake ≈ risk_budget / (stop_distance_fraction + fee_fraction + slippage_buffer)`. مقادیر واقعی از قواعد بازار، موجودی و شبیه‌سازی fill می‌آیند؛ این تقریب جایگزین محاسبهٔ دقیق/آزمون‌شده نیست.
4. **تعدیل اطمینان:** افزایش uncertainty، ضعف calibration، drift، تعارض مدل‌ها، نقدشوندگی پایین یا بدترشدن هزینهٔ اجرا باید بودجه/اندازه را کاهش دهد یا به تصمیم «عدم معامله» منجر شود. confidence بالا به‌تنهایی دلیل کافی برای افزایش اندازه نیست.
5. **تخصیص در سطح پرتفوی:** ریسک دارایی جدید با exposure و همبستگی موقعیت‌های موجود سنجیده می‌شود؛ این قابلیت در چندحسابی باید هم در سطح worker/account و هم در supervisor تجمیعی پرتفوی ارزیابی شود.

### قرارداد خروجی و اعتبارسنجی مستقل
هر پیشنهاد باید حداقل `risk_budget`, `suggested_stake`, `stop_price` یا `stop_distance`, `regime_id/probabilities`, `atr_snapshot`, `calibrated_confidence`, `uncertainty`, `expected_edge_net_costs`, `model_revision`, `decision_id` و زمان انقضا داشته باشد. Risk Gate بدون اعتماد کور به خروجی مدل، موجودی آزاد، حداقل/حداکثر و precision بازار، تازگی داده، سازگاری stop با API نسخهٔ pin شده، هزینه‌ها و ظرفیت باقی‌مانده تا سقف مطلق را دوباره محاسبه می‌کند. خروجی ناقص یا ناسازگار برابر با عدم اجازهٔ ورود است.

### معیارهای ارزیابی و محدودیت اجرا
- ارزیابی با walk-forward زمانی و OOS، تفکیک بر حسب regime/volatility، هزینهٔ معامله، stress tests و مقایسه با baselineهای ساده انجام شود.
- معیارها فقط بازده نیستند: maximum drawdown، tail loss/CVaR، توزیع زیان به‌ازای معامله، turnover، هزینه، calibration، stability بین foldها و کیفیت اجرای stop سنجیده شوند.
- باید با آزمون‌های کنترل‌شده ثابت شود که افزایش نوسان یا افت calibration باعث کوچک‌شدن اندازه/افزایش محافظه‌کاری می‌شود و مدل در شرایط OOD به‌طور پیش‌فرض معامله را متوقف می‌کند.
- stop-loss بازار، تضمین اجرای دقیق قیمت نیست؛ gap، slippage، قطعی اتصال و نقدشوندگی می‌توانند زیان واقعی را از زیان مدل‌شده بیشتر کنند. به همین دلیل سقف مطلق و supervisor مستقل باقی می‌مانند.
- تا قبل از گذر از آزمون‌های OOS و fault injection، خروجی Dynamic Risk Allocation صرفاً shadow/paper است و اجازهٔ تغییر اندازهٔ live ندارد.

## ۵.۳ سقف‌های محافظتی مطلق و قواعد اجرایی
- مالک پروژه در فاز صفر **فقط سقف محافظتی مطلق سرمایه** را تعیین می‌کند؛ برای مثال حداکثر افت کل equity یا یک آستانهٔ صریح Kill Switch مانند تعداد معینی زیان متوالی. انتخاب metric، مقدار، مرجع محاسبه و رفتار پس از فعال‌شدن باید صریح و نسخه‌بندی‌شده باشد.
- مدل هیچ اختیار نوشتن/تغییر این سقف، وضعیت latched مربوط به Kill Switch یا policy store را ندارد. رسیدن به سقف، ورود جدید را متوقف و سامانه را تا re-arm مجاز در حالت halt نگه می‌دارد؛ مدیریت موقعیت باز طبق runbook اضطراری آزموده‌شده انجام می‌شود.
- عدد ثابت مالک برای daily loss، اندازهٔ stake، اندازهٔ position، ATR multiplier یا stop-loss لازم نیست. این‌ها از خروجی پویا به دست می‌آیند، اما همگی مشمول کنترل مستقل و سقف محافظتی مطلق هستند.
- قواعد فنی غیرقابل‌تخطی همچنان برقرارند: Spot-only در نسخهٔ اول؛ عدم استفاده از leverage/margin/futures؛ عدم سفارش بیش از موجودی آزاد یا محدودیت بازار؛ allowlist؛ precision/minimum order؛ stale/gap gate؛ کنترل slippage، rate limit و وضعیت نامعلوم سفارش.
- پس از قطع اتصال/ابهام order state، ارسال مجدد تا reconciliation ممنوع است؛ idempotency/client order id در صورت پشتیبانی استفاده شود.
- کلیدهای API بدون مجوز برداشت، با IP allowlist در صورت پشتیبانی، و کلید مجزا برای هر حساب/محیط نگهداری شوند.

## ۵.۴ محل اعمال کنترل‌ها
1. **Policy store:** فقط سقف محافظتی مطلق، semantics و status مربوط به halt در policy نسخه‌بندی‌شده و خارج از دسترس نوشتن مدل نگهداری می‌شود.
2. **Dynamic Risk Allocation در worker:** پیش از ارسال پیشنهاد stake و stop را با رژیم، ATR، اطمینان کالیبره‌شده و شرایط جاری محاسبه می‌کند؛ خروجی برای هر decision ثبت می‌شود.
3. **Risk Gate در worker:** قبل از ورود از `confirm_trade_entry` و callbackهای متناسب Freqtrade، با `custom_stake_amount` و سازوکار stop پشتیبانی‌شدهٔ نسخهٔ هدف، پیشنهاد پویا را با موجودی، قیمت، هزینه و سقف باقی‌ماندهٔ مجاز می‌سنجد. callbackهای دقیق را با نسخهٔ pin شده تطبیق دهید.
4. **Freqtrade protections و stop-loss:** Protectionهای پشتیبانی‌شده (مثلاً StoplossGuard و MaxDrawdown) و stop-loss داخلی به‌عنوان دفاع تکمیلی استفاده شوند، نه به‌عنوان جانشین Dynamic Risk Allocation یا سقف محافظتی مطلق. در backtesting فعال‌سازی Protectionها طبق مستندات انجام شود. [R4]
5. **Risk Supervisor بیرونی:** equity/PnL/order status را مستقل پایش می‌کند؛ در فعال‌شدن سقف مطلق، allow-entry را لغو، سفارش‌های pending را بررسی و worker را halt می‌کند. مدیریت موقعیت‌های باز طبق runbook امن انجام می‌شود؛ shutdown ساده معادل لغو تضمین‌شدهٔ همهٔ سفارش‌های صرافی نیست.
6. **Deployment:** مدل به filesystem policy/secret دسترسی نوشتن ندارد؛ worker در کانتینر با read-only mounts برای policy و حداقل دسترسی شبکه اجرا می‌شود. فقط فرایند مجاز secret را می‌خواند.
7. **Release gate:** اگر supervisor، Ledger یا reconciliation قابل‌دسترس نیست، ورود جدید ممنوع؛ خروج/کاهش ریسک تا حدی که مسیر عملیاتی امن دارد ادامه می‌یابد.

## ۵.۵ حدود تضمین و تهدید
این معماری، مدل AI را از اختیار مستقیم بر کلید و سیاست ریسک محروم می‌کند و مسیر منطقی سفارش را از gate عبور می‌دهد. با این حال **کد استراتژی و worker همچنان در مرز اعتماد نرم‌افزاری هستند**؛ اگر کل worker یا محیط اجرایی compromise شود، هیچ شرطی در یک سند معماری نمی‌تواند تضمین مطلق بدهد. برای تهدید compromise کامل باید جداسازی فرایند، دسترسی محدود به secret، مراقبت از supply chain، محدودیت API و در صورت نیاز واسطهٔ اجرای سفارشِ policy-enforcing مستقل طراحی و آزموده شود. ساخت چنین واسطه‌ای خود تصمیم معماری جداست و نباید به‌صورت پنهانی یک موتور معاملهٔ موازی ایجاد کند.

## ۵.۶ رفتارهای fail-closed
| وضعیت | رفتار لازم |
|---|---|
| دادهٔ stale/gapped/invalid | رد ورود؛ ثبت reason code |
| مدل غایب، قدیمی یا schema mismatch | رد ورود؛ no fallback به مدل نامعلوم |
| NLP down | در حالت پیش‌فرض رد ورود؛ استثنا فقط با policy ارزیابی‌شده |
| Risk Supervisor / ledger unavailable | توقف ورود جدید |
| فعال‌شدن سقف محافظتی مطلق / Kill trigger | Kill Switch latched، توقف ورود، بررسی pending orders و اجرای runbook اضطراری |
| order status نامعلوم | توقف retry سفارش، reconciliation با صرافی |
| ناسازگاری balance/trades | توقف ورود و نیاز به reconciliation/تأیید عملیاتی |
| timeout یا exception پیش‌بینی | هیچ سیگنال مثبت پیش‌فرضی ایجاد نشود |

# ۶. داده‌ها و Decision Ledger
## ۶.۱ منابع حقیقت
- **Freqtrade trade DB**: منبع رسمی چرخهٔ معامله و state داخلی worker.
- **Exchange account**: منبع حقیقت موجودی و وضعیت نهایی سفارش/پرشدن؛ در restart با آن reconciliation شود.
- **Market store**: OHLCV و داده‌های خام قابل‌بازتولید، با منبع، timeframe و snapshot fingerprint.
- **News store**: دادهٔ خبری خام/metadata و خروجی NLP با زمان‌های point-in-time و مجوز retention.
- **Decision Ledger**: ردپای append-only برای چرایی پیشنهاد و رأی ریسک؛ منبع جایگزین وضعیت سفارش نیست.

## ۶.۲ حداقل رویدادهای Ledger
`decision_id`, `correlation_id`, `tenant_id`, `account_id`, `exchange_id`, `strategy_version`, `model_id/revision`, `feature_set_version`, `market_window_fingerprint`, `data_quality_status/reasons`, `news_event_ids`, `decision_time`, `action_proposed`, `probabilities/uncertainty`, `expected_edge`, `suggested_stake`, `risk_policy_version`, `risk_vote`, `risk_reasons`, `order_id/client_order_id`, `fills`, `fees`, `realized_outcome`, `latency_ms`, `error_code`.

- هر رویداد decision قبل یا هم‌زمان با اقدام سفارش ثبت شود؛ در صورت عدم امکان ثبت Ledger و طبق policy fail-closed، ورود جدید انجام نشود.
- Ledger باید append-only یا دارای کنترل integrity باشد؛ اصلاح رکورد با event اصلاحی انجام شود، نه overwrite بی‌ردپا.
- از درج API key، token، ایمیل/PII غیرضروری یا متن حساس خبر در log خودداری شود. دسترسی به ledger بر اساس tenant محدود گردد.
- نگهداری متن کامل خبر تابع terms منبع و retention policy است؛ برای audit، hash و metadata ممکن است کافی باشد.

## ۶.۳ پاسخ عملیاتی به «چرا خرید کرد؟»
با `decision_id` بتوان feature snapshot، نسخهٔ مدل، زمان خبر، پیش‌بینی primary، MetaLabel، رأی Risk Gate، پارامترهای اجرای strategy، سفارش/پرشدن و نتیجهٔ پس از معامله را بازیابی کرد. توضیح باید به شواهد ثبت‌شده متکی باشد، نه خلاصه‌ای که مدل پس از واقعه ساخته است.

## ۶.۴ بازیابی پس از crash
1. supervisor ابتدا ورود جدید را متوقف می‌کند و worker را در حالت recovery نگه می‌دارد.
2. worker جدید همان account/exchange را با حجم داده و artifact نسخه‌بندی‌شده راه می‌اندازد؛ هم‌زمان دو worker برای یک حساب اجازهٔ نوشتن سفارش ندارند.
3. database محلی، باز بودن سفارش‌ها، معاملات، fills و balance از صرافی خوانده و با state داخلی Freqtrade تطبیق داده می‌شوند.
4. هر سفارش/معاملهٔ orphan یا اختلاف موجودی به quarantine عملیاتی می‌رود؛ ارسال سفارش جایگزین تا حل اختلاف ممنوع است.
5. history کوتاه‌مدت بازار، model manifest و feature schema تطبیق داده می‌شوند. اگر مدل/feature artifact قابل بازیابی نیست، prediction worker آمادهٔ معامله نمی‌شود.
6. پس از reconciliation موفق، سلامت داده و Risk Supervisor بررسی و فقط سپس worker می‌تواند به حالت dry-run/live مجاز منتقل شود.
7. Kill Switch به‌طور خودکار با restart پاک نمی‌شود؛ بازکردن آن نیازمند مجوز مستقل و ثبت reason است.

# ۷. امنیت، قابلیت مشاهده و عملیات
## ۷.۱ امنیت secrets
- کلیدهای API فقط مجوز Spot trade لازم، بدون withdrawal، و ترجیحاً IP restricted داشته باشند.
- secret manager یا secret mount مجزا برای هر worker؛ secrets در environment dump، exception، config commit یا model prompt قرار نگیرند.
- مدل، tokenizer و فایل دادهٔ ورودی را untrusted بدانید؛ دانلود مدل فقط در مرحلهٔ build بررسی‌شده و از revision ثابت انجام شود.
- dependency lock، hash artifact، vulnerability scan و provenance برای هر release لازم‌اند.
- AI/NLP در سطح سیستم‌عامل دسترسی shell، package installation یا network egress نامحدود ندارد.

## ۷.۲ مشاهده‌پذیری
Metrics اصلی: تازه‌بودن کندل/خبر، تعداد gapها، نرخ رد Data Quality Gate، latency هر مدل، درصد predictionهای missing، کالیبراسیون مدل، risk reject rate، سفارش‌های pending، اختلاف DB با exchange، realized/unrealized PnL، drawdown، retry/error rate، و وضعیت kill switch.

Alertها باید به تفکیک tenant/account/exchange باشند. هر alert باید شدت، timestamp، reason code و runbook مشخص داشته باشد. مدل نباید بتواند alert را خاموش کند.

## ۷.۳ مدیریت مدل
- model artifact، dataset snapshot، training parameters، feature schema، target horizon، metrics و revisionها در Model Registry ثبت شوند.
- promotion از `research` به `validated`، سپس `paper` و `live-approved` مراحل جدا با امضای release است.
- drift باعث هشدار/بررسی و احتمالا توقف ورود می‌شود؛ retrain خودکار به معنای rollout خودکار نیست.
- rollback به artifact قبلی فقط پس از schema compatibility و تست restart/reload مجاز است.

# ۸. ADR — دفتر تصمیمات معماری
## ADR-001: Freqtrade به‌عنوان موتور معامله
**وضعیت:** پذیرفته‌شده. **تصمیم:** حفظ Freqtrade برای order lifecycle، backtesting، exchange integration و run mode. **دلیل:** استفاده از چارچوب آماده و پرهیز از بازنویسی بخش‌های حساس. **هزینه:** version pin و آزمون سازگاری APIهای داخلی لازم است؛ توسعه باید از extension pointها انجام شود. [R1][R3]

## ADR-002: استفادهٔ مجدد از آداپتور Spot نوبیتکس در Novaryn
**وضعیت:** پذیرفته‌شده به‌عنوان نقطهٔ شروع؛ ادغام runtime مشروط. **تصمیم:** از کد موجود reuse کنیم، نه اینکه آداپتور از صفر نوشته شود. ابتدا repo URL، license/ownership، commit و تست‌های Novaryn ثبت می‌شوند؛ سپس آن را به مرز exchange/CCXT مورد قبول Freqtrade می‌رسانیم. **گزینهٔ ترجیحی:** اگر آداپتور از قبل CCXT-compatible است و Freqtrade آن را با روش پشتیبانی‌شده instantiate می‌کند، همان مسیر به کار رود. در غیر این صورت، لایهٔ API/auth/normalization/rate-limit موجود حفظ و تنها پوستهٔ سازگاری اضافه می‌شود. اگر Freqtrade برای exchange ID سفارشی تغییر داخلی لازم داشته باشد، ابتدا upstream PR یا extension رسمی بررسی می‌شود؛ fork pinشده فقط پس از ADR جدا و تحلیل هزینهٔ نگهداری مجاز است. **ممنوع:** یک executor مستقل از Freqtrade که همان حساب را هم‌زمان معامله کند. [R3][R5][R8]

**تست‌های آداپتور:** market listing و symbol mapping، USDT quote validation، balances، precision/min/max، fees، order types supported، partial fill، cancel/status، retry/rate-limit, timestamp/clock, idempotency، unknown order, network failure و reconciliation. جفت‌های USDT فقط از market metadata واقعی enable می‌شوند؛ وجود مسیر API به‌تنهایی اثبات پشتیبانی همهٔ endpointها نیست.

## ADR-003: DeepAlpha اختیاری با Plan B بومی FreqAI
**وضعیت:** تصمیم پذیرفته‌شده، بسته مشروط. **تصمیم:** PoC ایزوله انجام شود؛ پذیرش فقط بر اساس سازگاری نسخه و no-core-patch. اگر شکست خورد، Triple-Barrier، OOF Primary Signals، Meta-Labeling و purged walk-forward به‌صورت مدل ترکیبی سفارشی FreqAI توسعه می‌یابد. **دلیل:** عدم وابستگی معماری به یک پروژهٔ ثالث و حفظ نسخه‌پذیری. [R1][R6][R7]

## ADR-004: Hugging Face محلی برای NLP، با مجوز و آزمون مدل
**وضعیت:** پذیرفته‌شده مشروط. **تصمیم:** inference محلی برای تأخیر و کنترل داده؛ مدل‌ها با revision ثابت و license manifest انتخاب می‌شوند. دو نامزد اولیه شامل Kenpache multilingual financial sentiment و AnkitAI DistilBERT financial news هستند؛ اما مجوز dataset AnkitAI باید روشن شود. CardiffNLP به‌عنوان benchmark مکمل برای متن کوتاه پیشنهاد می‌شود. **دلیل:** پلتفرم مشترک مدل و قابلیت اجرای محلی. **ریسک:** مدل‌های عمومی با اخبار واقعی رمزارز ممکن است domain shift داشته باشند؛ label و confidence نیاز به calibration دارند. [R9][R10][R11][R12]

## ADR-005: جداسازی worker برای multi-tenancy
**وضعیت:** پذیرفته‌شده. **تصمیم:** یک worker مستقل برای هر tuple حساب/صرافی/محیط؛ Control Plane مرکزی، Risk Supervisor در سطح account و portfolio. **دلیل:** جداسازی secret، DB و state، و محدود کردن blast radius. **هزینه:** مصرف منابع و عملیات بیشتر، که با template/automation کنترل می‌شود.

## ADR-006: Data Quality Gate مستقل
**وضعیت:** پذیرفته‌شده. **تصمیم:** gate پیش از feature generation و inference، با reason code و رفتار reject صریح. **دلیل:** مدل نمی‌تواند کیفیت داده را تضمین کند و اصلاح خاموش داده در backtest خطر leakage/false confidence دارد.

## ADR-007: Risk Supervisor جدا از مدل و دروازهٔ release
**وضعیت:** پذیرفته‌شده. **تصمیم:** سقف محافظتی مطلق و Kill Switch توسط policy نسخه‌بندی‌شده و supervisor مستقل اجرا شوند؛ بودجهٔ روزانه، stake و stop به‌صورت پویا توسط Dynamic Risk Allocation پیشنهاد می‌شوند و Risk Gate آن‌ها را محدود به ظرفیت باقی‌مانده می‌کند. Protectionهای Freqtrade لایهٔ دفاعی مکمل‌اند. **دلیل:** بودجهٔ پویا نباید با سقف مطلق اشتباه شود و وضعیت‌های crash/unknown order به کنترل بیرونی نیاز دارند. [R4]

## ADR-008: Ledger مستقل و reconciliation اجباری
**وضعیت:** پذیرفته‌شده. **تصمیم:** Ledger append-only در کنار DB Freqtrade و reconciliation با exchange پس از crash. **دلیل:** DB داخلی به‌تنهایی توضیح کامل feature/model/risk vote را ثبت نمی‌کند؛ Ledger نیز جایگزین exchange state نیست.

## ADR-009: Dynamic Risk Allocation vs. Hard Limits
**وضعیت:** پذیرفته‌شده برای معماری؛ اجرای زنده مشروط به اعتبارسنجی OOS و fault injection.
**زمینه:** تعیین ثابت stake، stop-loss و بودجهٔ ریسک روزانه با هدف خودمختاری AI-driven سازگار نیست؛ اما واگذاری اختیار نامحدود به مدل می‌تواند امنیت سرمایه را تضعیف کند.
**تصمیم:** FreqAI/مدل رژیم/Meta-Labeling پیشنهاد پویای بودجهٔ روزانه، اندازهٔ موقعیت و stop را با توجه به رژیم بازار، ATR، confidence کالیبره‌شده، uncertainty، هزینهٔ اجرا، exposure و ظرفیت باقیمانده تولید می‌کنند. یک Risk Gate قطعی و Risk Supervisor مستقل پیشنهاد را دوباره اعتبارسنجی می‌کنند. مالک در فاز صفر سقف محافظتی مطلق را تعیین می‌کند؛ این سقف و Kill Switch immutable هستند و خارج از اختیار مدل نگهداری می‌شوند.
**قواعد لازم:** (۱) مدل حق تغییر policy یا خاموش‌کردن halt ندارد؛ (۲) خروجی ناقص/ناسازگار یا دادهٔ ناسالم به رد ورود می‌انجامد؛ (۳) افزایش نوسان/uncertainty یا افت calibration باید اندازه را کاهش دهد یا معامله را حذف کند؛ (۴) ریسک پیشنهادی باید شامل fees/slippage باشد؛ (۵) stop قیمت تضمین‌شدهٔ اجرا نیست و سقف مطلق همچنان لازم است.
**معیار پذیرش:** dynamic sizing/stop در walk-forward/OOS و آزمون‌های رژیم/volatility بررسی شود؛ fault injection فعال‌شدن سقف، crash، stale data، مدل unavailable و restart را پوشش دهد؛ هیچ حالت آزموده‌شده‌ای نباید امکان دورزدن سقف یا duplicate order ایجاد کند.
**پیامدها:** خودمختاری بیشتر در شرایط عادی، به بهای پیچیدگی مدل، calibration و اعتبارسنجی بیشتر. تا تصویب معیارهای پذیرش، خروجی allocator فقط در shadow/paper مجاز است.

## ADR-010: Connectivity Manager با routeهای مجاز و قابل ممیزی
**وضعیت:** پذیرفته‌شده از نظر معماری؛ جزئیات transport به آزمون نسخه/محیط وابسته است.
**تصمیم:** یک `Connectivity Manager` برای Direct، HTTP/HTTPS Proxy، SOCKS Proxy، VPN و SSH Tunnel ایجاد می‌شود. هر سرویس و محیط route policy جداگانه، health probe، timeout، retry budget، egress identity و telemetry دارد. Route failover فقط میان مسیرهای از پیش تأییدشده و منطبق با شرایط provider مجاز است.
**مرز امنیتی:** این لایه برای عبور از rate limit، جغرافیا/مجوز قراردادی، allowlist یا دسترسی غیرمجاز طراحی نمی‌شود. TLS verification خاموش نمی‌شود؛ secrets در Git نیستند؛ و پس از order timeout تا reconciliation تغییر مسیر/ارسال مجدد ممنوع است.
**معیار پذیرش:** ماتریس Direct/HTTP(S)/SOCKS/VPN/SSH برای endpointهای واقعی، شامل auth, TLS, egress IP, latency, reconnect و failover، pass/fail دارد. عدم پشتیبانی یک مسیر در نسخه/محیط هدف با دلیل ثبت می‌شود.

## ADR-011: External Data Provider Registry و جداسازی Feature Store از News Store
**وضعیت:** پذیرفته‌شده.
**تصمیم:** داده‌های market analytics/آنچین به External Feature Store و متن خبر/تقویم/social مجاز به News Store و NLP Service می‌روند. همهٔ feedها adapter مستقل، source provenance، timestampهای point-in-time، schema/rate limit policy و license status دارند. دادهٔ صرافی معامله‌گر منبع حقیقت order/fill/balance باقی می‌ماند.
**منابع کاندید:** CoinGecko، CoinMarketCap، CoinGlass، On-Chain APIs، Whale Alert، RSS News، Economic Calendar، X/Twitter API، Telegram connector مجاز و منابع خبری/تحلیلی فارسی با RSS/API مجاز. TradingView تا ارائهٔ مجوز صریح استفادهٔ non-display در تصمیم الگوریتمی، `blocked_for_ai_trading` است.
**معیار پذیرش:** هر provider در registry دارای endpoint رسمی، شرایط استفاده، tier/rate limits، retention/redistribution، freshness/SLO، مالک، تست contract و تصمیم `approved/conditional/blocked` است. تنها دادهٔ مجاز و قبول‌شده توسط Data Quality Gate می‌تواند به مدل برسد.

## ADR-012: Project Governance & Owner Authority
**وضعیت:** پذیرفته‌شده؛ اجرای فنی مشروط به change-control در GitHub.
**زمینه:** اسناد پیشین سلسله‌مراتب اختیارات مالک و روش برخورد با دستور متعارض را روشن نکرده بودند.
**تصمیم:** مالک مرجع نهایی تصمیم‌های داخلی پروژه است و می‌تواند scope، اولویت‌ها، معماری و سیاست‌های داخلی را با Owner Directive صریح تغییر دهد. دستور متعارض با سند فعلی، به تغییر نسخه‌بندی‌شدهٔ سند/ADR و ارزیابی اثر امنیت و سرمایه تبدیل می‌شود؛ تغییر بی‌ردپا یا bypass لحظه‌ای gate مجاز نیست.
**حد اجرا:** دستور فعال‌سازی live، درخواست آغاز Release Authorization است، نه دسترسی مستقیم به حساب. Runtime پس از عبور همهٔ gateها و تأییدهای الزامی runbook را اجرا می‌کند. ChatGPT به‌خودی‌خود دسترسی سیستم محلی/secret/صرافی ندارد و اجرای معامله را از گفتگو انجام نمی‌دهد. قوانین قابل‌اعمال، شرایط صرافی و کنترل‌های امنیتی پابرجا هستند.
**ردپای الزامی:** `directive_id`, timestamp، اسناد/ADRهای متأثر، ارزیابی ریسک/امنیت، تست‌ها، commit SHA، تصمیم GO/NO-GO و در live release/config/model hashes و rollback plan.
**معیار پذیرش:** تعارض مالک با سند داخلی منجر به issue/commit/نسخهٔ تازه و audit event شود؛ دستور چتی به‌تنهایی نتواند kill switch یا release gate را غیرفعال کند.

## ADR-013: Daily Reporting Engine and Bale Notifications
**وضعیت:** پذیرفته‌شده از نظر معماری؛ قرارداد API بله و SDK در فاز صفر/یک باید تأیید شوند.
**تصمیم:** Reporting Engine به‌عنوان سرویس read-only هر روز ساعت ۰۷:۰۰ در `reporting.timezone` snapshot reconciled می‌سازد، نمودارهای PnL/equity/trade/operations تولید می‌کند و از طریق `integrations/notifications/bale/` برای chat allowlist شده ارسال می‌کند. در نسخهٔ اول بله یک کانال اعلان یک‌طرفه است. مرجع پیاده‌سازی: مستندات رسمی Bale Bot API که پیوند آن در نقشهٔ راه و registry منابع نگهداری می‌شود.
**امنیت/دقت:** Bot token فقط در secret mount/manager است؛ چت غیرمجاز رد می‌شود؛ گزارش دارای report_id/source hashes/delivery state است؛ دادهٔ غیرreconciled واضح برچسب‌گذاری می‌شود؛ retry محدود و idempotency از ارسال تکراری جلوگیری می‌کند. هیچ فرمانی از بله سفارش، تغییر risk policy یا فعال‌سازی live نمی‌کند.
**گام پذیرش:** ابتدا mock/contract tests و گزارش مصنوعی؛ سپس با دادهٔ backtest/paper تطبیق PnL؛ در dry-run ارسال واقعی به chat آزمایشی مجازشده؛ live reporting تنها پس از release gate.
**معیار پذیرش:** زمان‌بندی 07:00، timezone، ارسال متن/تصویر/ضمیمه، خطاهای API، restart و redaction secrets آزموده‌اند؛ مجوز و نگهداری SDK/API در registry ثبت شده است.

# ۹. دروازه‌های انتشار نهایی
اجرای live فقط زمانی قابل بررسی است که همهٔ موارد زیر پذیرفته شوند:
- [ ] نسخهٔ Freqtrade/CCXT/Python و dependency lock ثابت و قابل بازتولید است.
- [ ] Novaryn adapter با URL و commit ثابت، license بررسی‌شده و contract tests سبز در محیط غیرواقعی است.
- [ ] بازارهای USDT روی صرافی هدف از metadata واقعی تأیید شده‌اند؛ جفت غیرموجود وارد whitelist نمی‌شود.
- [ ] Data Quality Gate gap/stale/schema/leakage tests را می‌گذراند.
- [ ] مدل primary و Meta-Label با purged walk-forward و OOS ارزیابی شده و baseline را بر اساس معیارهای از پیش تعیین‌شده می‌سنجد.
- [ ] مجوز وزن NLP، dataset، خبر و dependencyها ثبت و از نظر تجاری بررسی شده است.
- [ ] Dynamic Risk Allocation و Risk Supervisor، سقف محافظتی مطلق، Kill Switch و رفتار fail-closed در fault injection و سناریوهای رژیم/نوسان آزموده شده‌اند.
- [ ] بازیابی سفارش باز و crash بدون duplicate order تأیید شده است.
- [ ] workerها secret/DB/model writable directory مشترک ندارند.
- [ ] dry-run/paper نتایج و رفتار اجرا را برای دورهٔ مصوب نشان داده است.
- [ ] سرمایهٔ شروع، سقف محافظتی مطلق و برنامهٔ rollback توسط مالک به‌صورت کتبی تصویب شده‌اند؛ مقادیر stake/stop/بودجهٔ روزانه خروجی پویا هستند و به‌عنوان عدد ثابت از مالک درخواست نمی‌شوند.
- [ ] Owner Directive/Release Authorization، اثر ریسک/امنیت، commit و rollback ثبت شده‌اند؛ هیچ دستور محاوره‌ای به‌تنهایی gate را دور نمی‌زند.
- [ ] گزارش روزانهٔ گرافیکی از snapshot reconcileشده تولید و در Bale chat allowlist شده تحویل می‌شود؛ شکست ارسال/دادهٔ ناقص ثبت و اعلام می‌شود.

# منابع و مراجع فنی
منابع زیر برای طراحی و اعتبارسنجی نسخهٔ حاضر بررسی شده‌اند. وضعیت هر مدل، پکیج و مستند باید هنگام هر release مجدداً کنترل شود؛ مجوزها و APIهای ثالث ثابت فرض نمی‌شوند. تغییر نسخهٔ Freqtrade، CCXT، آداپتور Novaryn، مدل NLP، دیتاست، منبع خبر یا policy ریسک مستلزم بازبینی بخش‌ها و ADRهای مرتبط، اجرای مجدد تست‌های پذیرش و ثبت نسخهٔ جدید است.

- **[R1] Freqtrade — FreqAI Developer Documentation:** https://www.freqtrade.io/en/stable/freqai-developers/
- **[R2] Freqtrade — FreqAI Feature Engineering:** https://www.freqtrade.io/en/stable/freqai-feature-engineering/
- **[R3] Freqtrade — Exchange Documentation:** https://www.freqtrade.io/en/stable/exchanges/
- **[R4] Freqtrade — Protections / Plugins:** https://www.freqtrade.io/en/stable/plugins/
- **[R5] CCXT — Repository and exchange support:** https://github.com/ccxt/ccxt
- **[R6] PyPI — deepalpha-freqai:** https://pypi.org/project/deepalpha-freqai/
- **[R7] DeepAlpha repository:** https://github.com/stefanoviana/deepalpha
- **[R8] Nobitex official API documentation:** https://apidocs.nobitex.ir/
- **[R9] AnkitAI financial-news sentiment model:** https://huggingface.co/AnkitAI/distilbert-base-uncased-financial-news-sentiment-analysis
- **[R10] Kenpache multilingual financial sentiment v2:** https://huggingface.co/Kenpache/multilingual-financial-sentiment-v2
- **[R11] CardiffNLP Twitter RoBERTa sentiment latest:** https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
- **[R12] ProsusAI/FinBERT model page:** https://huggingface.co/ProsusAI/finbert
- **[R13] Hugging Face Transformers documentation:** https://huggingface.co/docs/transformers/index
- **[R14] Freqtrade source repository and license:** https://github.com/freqtrade/freqtrade
- **[R15] TradingView Terms/Policies (non-display restrictions):** https://www.tradingview.com/policies/
- **[R16] CoinGecko API pricing and terms entry point:** https://www.coingecko.com/en/api/pricing
- **[R17] CoinMarketCap API documentation:** https://coinmarketcap.com/api/documentation/
- **[R18] CoinGlass API endpoint documentation:** https://docs.coinglass.com/reference/endpoint-overview
- **[R19] Whale Alert API documentation:** https://developer.whale-alert.io/api-account/documentation
- **[R20] X API policy and data access:** https://help.x.com/en/rules-and-policies/x-api
- **[R21] Telegram APIs:** https://core.telegram.org/api
