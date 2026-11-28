سند مرجع جامع معماری، حاکمیت و نقشهٔ راه

Master Reference Document | نسخهٔ ۱٫۳

۲۸ نوامبر ۲۰۲۶

به‌روزرسانی بر مبنای ADR-050 تا ADR-076، Trial Registry تاریخی و شواهد GitHub تا ۲۸ نوامبر ۲۰۲۶؛ M7 رسماً CLOSED_NO_GO و M8.1 Capture و تحلیل اکتشافی BTCUSDT تکمیل شده است.

وضعیت جاری در MRD v1.3: M7 Cross-Sectional = CLOSED_NO_GO؛ corpus قدیمی M1.6 برای BTCUSDT همچنان BLOCKED_DATA_MARKET_MISMATCH است؛ M8.1 یک Capture عمومی BTCUSDT به مدت ۶۰٫۰۸۸ دقیقه انجام داده و تحلیل اکتشافی آن طبق ADR-076 با نتیجهٔ NO_SIGNIFICANT_OBI_CORRELATION تکمیل شده است. Trial Registry متمرکز: ۱۶ candidate identity و ۴۳ رکورد slot/identity در سطح خانواده‌های ثبت‌شده؛ M8.1 Capture یک Trial اقتصادی نیست؛ GLOBAL_N همچنان UNKNOWN. main دست‌نخورده مانده؛ HEAD=a1c1a658906cd79be62796885e3ed782b66a267d و Tree SHA=5761c6c34143f8e6924f53a4e745c48932a5f791. Holdout=SEALED و trading_authorization=NO-GO.

Champion = null   |   order_permission = false   |   trading_authorization = NO-GO

private_api_calls = 0   |   real_order_attempts = 0   |   Holdout = SEALED

مبنای حاکمیتی: آزادی در کشف، محدودیت در اختیار؛ هیچ‌یک از نتایج NO-GO یا سبز بودن CI مجوز معامله ایجاد نمی‌کند.

# بخش افزودهٔ نسخهٔ ۱٫۳ — وضعیت مرجع جاری، بسته‌شدن M7 و مسیر فعال M8.1

این بخش در تاریخ ۲۸ نوامبر ۲۰۲۶ تهیه شده و بر هر عبارت تاریخی یا متعارض نسخه‌های v1.2/v1.1 تقدم دارد. متن v1.2 در ادامه برای حفظ provenance نگهداری می‌شود؛ ارجاع‌های آن به گزینه‌های A/B یا OWNER DECISION PENDING دیگر وضعیت جاری نیستند.

## ۱. وضعیت اجرایی و مرجعیت مخزن

- شاخهٔ پژوهشی فعال: `research/nobitex-hf-strategy`.
- HEAD شاخه پیش از commit نهایی MRD v1.3: `2cfc327eaf5c1a66fa922152a99db3e7c6484717` (پس از تکمیل M8.1، ثبت ADR-076 و حذف marker موقت).
- `main` همچنان FROZEN است: HEAD=`a1c1a658906cd79be62796885e3ed782b66a267d`؛ Tree SHA=`5761c6c34143f8e6924f53a4e745c48932a5f791`. در این به‌روزرسانی هیچ تغییر یا commit روی `main` انجام نشده است.
- Standard CI روی commit اجرایی M8.1 با SHA=`a0b550f23071f8bf920b866771eb75cba30d8bf6`: Run #2721 / ID `46106438112`، completed / success. این نتیجه فقط سلامت مهندسی همان SHA را اثبات می‌کند و شاهد Edge اقتصادی نیست.
- Workflow Capture با Run ID `46106438168` روی همان SHA از مسیر موقت `push` و فایل `TRIGGER_M8` ایجاد شد. Job دریافت داده، اعتبارسنجی، ممیزی و آپلود Artifact همگی `SUCCESS` شدند؛ Workflow کلی در مرحلهٔ ساخت ADR-076 به‌علت import ناقص `os` در اسکریپت گزارش‌نویسی شکست خورد. این نقص در مسیر بازتحلیل مستقل اصلاح شد. marker و تریگر موقت حذف شدند؛ Workflowهای جاری فقط `workflow_dispatch` را حفظ می‌کنند.
- Preflight عمومی و بدون احراز هویت `/v3/orderbook/BTCUSDT` موفق شد. Capture نهایی `BTCUSDT` را در manifest و کانال‌های WebSocket ثبت کرده است: ۲٬۵۷۹ snapshot دفتر سفارش، ۹۳ معامله، صفر gap؛ مدت واقعی ۶۰٫۰۸۸ دقیقه. Artifact خام ID `15249319923` و digest آن `sha256:bc16983be51490cafebff11c5de06360d6668633bd3a75ea1afbe0defd3837c0` است.

## ۲. وضعیت مسیرهای پژوهشی

| مسیر | حکم جاری | خلاصهٔ شواهد و مرز تفسیر |
|---|---|---|
| HGBQ-CQR / Regime P | CLOSED — NO-GO | ۳ candidate یکتا؛ ۳۰ slot شامل ۲۷ duplicate؛ هیچ Lower Bound از Cost-50/90 عبور نکرد. |
| M3 OFI / Regime M | CLOSED — NO-GO | ۲ candidate منطقی؛ دادهٔ تاریخی پنج segment از BTCIRT بود، نه BTCUSDT؛ هر دو نتیجهٔ اقتصادی ناموفق. |
| Regime T Triple-Barrier | CLOSED — NO-GO | ADR-056؛ Accuracy=51.09% در برابر آستانهٔ >55%. |
| M5.2 Sentiment | CLOSED — NO-GO | ۴ candidate؛ Sentiment در هر چهار paired comparison از Control بدتر بود. |
| M6 Three-Class Triple-Barrier | CLOSED — INSUFFICIENT_SAMPLE / NO-GO | ۲ candidate؛ هر دو ۰/۴ fold پایدار داشتند و NetPF هر دو افق زیر ۱ بود. |
| M7 Cross-Sectional Ridge | CLOSED_NO_GO | G2.0 به‌دلیل panel خالی و G3.0 به‌دلیل شکست Gateهای اقتصادی/پایداری بسته شدند؛ هیچ retry یا sweep دیگری مجاز نیست. |
| M8 Microstructure baseline | BLOCKED | پنج Artifact قدیمی M1.6 همگی BTCIRT هستند؛ برای فرضیه BTCUSDT قابل استفاده نیستند؛ همبستگی‌ها در آن corpus `NOT_COMPUTED` باقی می‌مانند. |
| M8.1 BTCUSDT Capture + correlation | COMPLETE — EXPLORATORY / NO_SIGNIFICANT_OBI_CORRELATION | Capture و hash audit موفق؛ تحلیل دوباره از Artifact immutable در Run `46113673381` موفق؛ OBI در ۵m معنادار نشد (N=8)، در ۱۵m فقط N=2 و correlation estimable نیست. این نتیجه NO-GO اقتصادی نیست. |

## ۳. M7 — تصمیم نهایی و Trial Registry

- حکم رسمی: `CLOSED_NO_GO` طبق ADR-074. هیچ feature sweep، تغییر مدل، تغییر universe، threshold search یا اجرای مجدد M7 زیر همین lineage مجاز نیست.
- G1.2: دو identity تاریخی `NOBITEX-M7-5PAIR-RIDGE-TOP2-01` و `NOBITEX-M7-10PAIR-RIDGE-TOP2-01` به علت `BLOCKED_DATA` پیش از Fit باقی مانده‌اند.
- G2.0: دو identity superseding عبارت‌اند از `NOBITEX-M7-5PAIR-RIDGE-TOP2-02` و `NOBITEX-M7-9PAIR-RIDGE-TOP2-01`. هر دو به `BLOCKED_DATA_OR_FEATURE_QUALITY` با علت `m7_panel_empty_after_feature_qualification` رسیدند؛ هیچ Ridge fit یا economic fold انجام نشد.
- G3.0: همان دو identity G2.0 با revision ویژگی `log_return_24h` به‌جای `relative_volume_shock_24h` ارزیابی شدند؛ این تغییر feature به‌عنوان configuration revision مستقل برای multiple-testing حفظ می‌شود، نه یک candidate identity جدید.
- G3.0 نتیجه: هر دو candidate و هر دو سناریوی 50/90 bps، NetPF<1 و MaxDD نزدیک 100% داشتند؛ هر دو ۰/۴ fold پایدار. 5PAIR: NetPF=0.503741 / 0.262960 و MaxDD=99.997453% / 99.9999997%؛ 9PAIR: NetPF=0.424442 / 0.193591 و MaxDD=99.999872% / 99.999999999%. هیچ Gate اقتصادی عبور نکرد.
- حسابداری M7: چهار هویت candidate ثبت‌شده در G1.2/G2.0؛ دو identity از آن‌ها در G3.0 با revision پیکربندی دوباره ارزیابی شدند. G3.0 یک نسخهٔ رایگان یا حذف‌شدنی از سابقه نیست و باید در multiple-testing lineage بماند.

## ۴. Trial Registry — شمارش اصلاح‌شده

- دوازده identity منطقی متمرکز پیش از M7 در پنج workstream ثبت شده‌اند: HGBQ-CQR=3، M3 OFI=2، Regime T=1، M5.2=4، M6=2.
- M7 چهار identity دیگر در revisions G1.2/G2.0 ثبت کرده است. بنابراین شمارش متمرکز candidate identityهای متمایز ثبت‌شده و ردشده برابر **۱۶** است، نه ۱۰ نسخهٔ قبلی.
- شمارش رکوردهای slot/identity در inventory متمرکز: ۳۰ slot تاریخی HGBQ (شامل ۲۷ duplicate) + ۲ OFI + ۱ Regime T + ۴ M5.2 + ۲ M6 + ۴ M7 = **۴۳ رکورد**. این جمع فقط برای inventory توصیفی است و معادل GLOBAL_N نیست.
- G3.0 M7 یک configuration revision روی همان دو identity G2.0 است و در multiple-testing lineage حفظ می‌شود؛ اما به‌عنوان دو identity منطقی تازه به شمارش یکتا اضافه نمی‌شود.
- `GLOBAL_N=UNKNOWN` و `GLOBAL_DSR=NOT_COMPUTABLE` بدون تغییر باقی می‌مانند. تعداد اجرای Workflow، attempt، candidate identity و config revision نباید با یکدیگر جمع یا یکسان فرض شوند.
- M8/M8.1 هنوز candidate اقتصادی ثبت‌شده ندارند. Capture داده، Trial اقتصادی نیست؛ تا عبور Source Gate و ثبت فرضیه/تحلیل، M8.1 به شمارش ۱۶ هویت افزوده نمی‌شود.

### ۵. نقشهٔ راه جاری — M8.1 به‌عنوان تنها مسیر فعال

۱) Capture عمومی BTCUSDT به مدت واقعی ۶۰٫۰۸۸ دقیقه در GitHub Actions تکمیل شد؛ هیچ هزینه، TinyFish، private API، credential یا سفارش استفاده نشد.
۲) `market=BTCUSDT`، کانال‌های عمومی REST/WebSocket، timestamps، manifest و وضعیت gap ممیزی شدند؛ ۲٬۵۷۹ snapshot دفتر سفارش، ۹۳ معامله و صفر gap ثبت شد.
۳) Artifact خام ID `15249319923` با digest `sha256:bc16983be51490cafebff11c5de06360d6668633bd3a75ea1afbe0defd3837c0` immutable آپلود شد؛ Artifact در Job جدا دانلود و manifest/CSV SHAها دوباره ممیزی شدند.
۴) Analyzer سبک `analyze_m8_microstructure.py` اجرا شد؛ OBI top-5 در gridهای UTC غیرهمپوشان، بازده آتی mid-price در ۵ و ۱۵ دقیقه، Pearson/Spearman و اصلاح Bonferroni بررسی شدند.
۵) ADR-076 در `docs/adr/ADR-076-m8-1-btcusdt-capture-and-correlation.md` ثبت شد؛ Run بازتحلیل `46113673381` و Artifact تحلیل `15250428572` هستند. نتیجه فقط اکتشافی است: ۵m با N=8 معنادار نشد؛ ۱۵m با N=2 از نظر correlation قابل برآورد نیست.
۶) هیچ DeepLOB، مدل ML پیچیده، Fit، tuning، آزمون سودآوری یا trading execution انجام نشد. Capture یک‌ساعته حداکثر حدود ۱۲ مشاهدهٔ غیرهمپوشان ۵دقیقه‌ای و ۴ مشاهدهٔ ۱۵دقیقه‌ای می‌دهد؛ این داده برای ادعای تأییدی کافی نیست.

## ۶. قواعد ایمنی و حدود دسترسی

`Champion = null` | `order_permission = false` | `trading_authorization = NO-GO`

`HOLDOUT_ACCESS = FORBIDDEN` | `Holdout = SEALED` | `posthack_data_read = false`

`private_api_calls = 0` | `real_order_attempts = 0` | `paper_orders = 0`

`ALIX_DRY_RUN = true` | `ALIX_ALLOW_LIVE_TRADING = false`

- Capture فقط از public REST/WebSocket endpoints مجاز انجام می‌شود؛ Authorization header و WebSocket connection token ارسال نمی‌شود.
- هیچ هزینه‌ای پرداخت نمی‌شود؛ هیچ provider پولی، TinyFish یا ابزار واسطه‌ای مجاز نیست.
- `main` FROZEN می‌ماند؛ هرگونه تغییر فقط روی `research/nobitex-hf-strategy` با commit محدود و compare-and-swap انجام می‌شود.
- Fail-closed: نبود دادهٔ معتبر، Artifact قابل‌ردیابی، Gate سبز یا مجوز معتبر به توقف منجر می‌شود؛ هرگز به دادهٔ BTCIRT برای فرضیه BTCUSDT تنزل/تبدیل نمی‌شود.

## ۷. منابع تصمیم و شواهد افزوده‌شده

- ADR-068 / ADR-069: دو Trial از M6 و نتیجهٔ `INSUFFICIENT_SAMPLE / NO-GO`؛ جزئیات دقیق در Trial Registry.
- ADR-070 تا ADR-074: انتخاب، پیش‌ثبت، G2.0/G3.0 و بستن رسمی M7.
- ADR-075: M1.6 Market Mismatch؛ داده‌های قدیمی BTCIRT و عدم محاسبهٔ همبستگی BTCUSDT.
- GitHub Actions Run `46106438112`: Standard CI success روی SHA اجرای M8.1 `a0b550f23071f8bf920b866771eb75cba30d8bf6`.
- GitHub Actions Run `46106438168`: Job Capture با نتیجهٔ `SUCCESS` و Artifact خام ID `15249319923`.
- GitHub Actions Run `46113673381`: بازتحلیل Artifact immutable، ممیزی SHAها، اجرای Analyzer، commit ADR-076 و بارگذاری Artifact تحلیل؛ کل Run با نتیجهٔ `SUCCESS`.
- ADR-076: `NO_SIGNIFICANT_OBI_CORRELATION`؛ ۵m: N=8، Pearson r=0.47108390 و Bonferroni p=1.0؛ ۱۵m: N=2 و Pearson/Spearman قابل‌برآورد نیستند.
- Artifact تحلیل ID `15250428572`، digest `sha256:cd8fa1b0f1cd14c6d9f27658e631838083d6f63065f17c41b36246c6070fd38d`؛ تحلیل اکتشافی و تحت محدودیت نمونه.

تأکید: وضعیت جاری این بخش جایگزین نقشهٔ راه تاریخی v1.2 است. هر مسیر بسته‌شده فقط در صورت فرضیهٔ واقعاً متمایز، preregistration جدید و مجوز مستقل مالک قابل بررسی است؛ هیچ نتیجهٔ NO-GO با تغییر نام زنده نمی‌شود.

---

# بخش تاریخی نسخهٔ ۱٫۲ — برای حفظ provenance؛ وضعیت جاری توسط نسخهٔ ۱٫۳ جایگزین شده است


این متن snapshot تاریخی v1.2 در ۲۸ نوامبر ۲۰۲۶ است. برای حفظ provenance بدون بازنویسی تاریخچه نگهداری شده و از تاریخ ۲۸ نوامبر ۲۰۲۶، افزودهٔ نسخهٔ ۱٫۳ در ابتدای سند مرجع جاری است.

### ۱. وضعیت مخزن و شواهد کیفیت در snapshot تاریخی v1.2

مورد

مقدار تأییدشده

شاخهٔ فعال

research/nobitex-hf-strategy

HEAD پایهٔ منبع برای v1.2

95133203e70265ac96f7d4dd7c14ed213fadaf38 (قبل از commit مستندات v1.2 و ADR-065)

Standard CI روی HEAD پایه

Run #46001067175 — completed / success؛ 1,127 passed؛ 5 skipped؛ coverage=90.36%

Four-Arm Quality Preflight

Run #45998903704 — success روی همان SHA؛ این شاهد فقط کیفیت مهندسی است، نه اقتصاد مدل.

main

HEAD=a1c1a658906cd79be62796885e3ed782b66a267d؛ Tree SHA=5761c6c34143f8e6924f53a4e745c48932a5f791. در این نوبت هیچ commit یا تغییر روی main انجام نشده است.

وضعیت تاریخی main

در فاز Dispatch قبلی، یک فایل Workflow موقت add و delete شد؛ Tree دقیقاً به baseline بازگشت اما دو commit در تاریخچهٔ main باقی ماندند. این نکته از «تاریخچهٔ بدون تغییر» متمایز است.

قید تفسیر: Standard CI فقط سلامت مهندسی همان SHA و jobهای اجراشده را تأیید می‌کند؛ موفقیت CI هیچ شاهدی برای لبهٔ اقتصادی، readiness، Paper یا Live نیست.

## ۲. مسیرهای بسته‌شده و حسابداری Trialها

حکم جاری سه مسیر مورد اشارهٔ مالک، صریحاً CLOSED / NO-GO است. Regime T Triple-Barrier نیز پیش‌تر مستقل از آن‌ها آزموده و NO-GO شده است؛ بنابراین گزینه A به شکل فعلی یک آزمایش تازه محسوب نمی‌شود.

مسیر / ADR

Trialهای ارزیابی‌شده و مرز شمارش

شواهد اصلی

حکم

HGBQ-CQR / Regime P
ADR-053 تا 055

۳۰ slot ثبت‌شده؛ ۳ candidate یکتا (سه horizon) + ۲۷ duplicate. G4.5/G4.6 ادامهٔ همان lineage هستند.

۰ Lower Bound بالاتر از Cost-50/90؛ ۰ trade. G4.6: ۹۰/۹۰ diagnostic با G4.5 دقیقاً برابر؛ حذف Volume بهبود نداد.

CLOSED NO-GO؛ ۰/۳

M3 OFI / Regime M
ADR-059/060

۲ Trial (1h و 4h)؛ فقط پنج segment از BTCIRT، نه BTCUSDT.

1h: 665 cohort، PF=0.07347، PnL=-7.9225%. 4h: 82 cohort، PF=0.23269، PnL=-0.1214%؛ stability=0/4.

CLOSED_NO_GO؛ ۰/۲

Regime T Triple-Barrier
ADR-056

۱ experiment؛ ۳ fold؛ 31,801 input؛ 24,740 binary events؛ 7,061 censored؛ 1,652 OOF.

Accuracy=51.09%؛ balanced=51.57%؛ gate مالک >55%. TIMEOUT از binary fit حذف شد؛ این run اقتصادی نبود.

CLOSED NO-GO؛ ۰/۱

M5.2 Sentiment
ADR-062 تا 064

۴ Trial: Control/Sentiment × 1h/4h؛ ۱۶ fold fit؛ 101,648 OOS rows.

همهٔ ۸ NetPF<1؛ stability=0/4؛ Sentiment در هر ۴ pair بدتر از Control بود.

CLOSED NO-GO؛ ۰/۴

جمع‌بندی شمارش در snapshot v1.2: ۱۰ identity (HGBQ=3، OFI=2، Regime T=1، M5.2=4) و ۳۷ slot. این عدد تاریخی است و توسط شمارش v1.3 در ابتدای سند supersede شده است؛ v1.3 با احتساب M6=2 و M7=4، تعداد متمرکز را به ۱۶ identity و ۴۳ رکورد slot/identity می‌رساند. G3.0 M7 یک configuration revision بر دو identity G2.0 است و identity تازه‌ای نمی‌سازد. GLOBAL_N=UNKNOWN و GLOBAL_DSR=NOT_COMPUTABLE.

## ۳. گزینهٔ A — بازطراحی Target با دادهٔ OHLCV موجود

مزیت بالقوه: استفاده از دادهٔ قبلی Nobitex BTCUSDT، بدون خرید منبع جدید؛ Logistic Regression ساده و قابل ممیزی باقی می‌ماند؛ تمرکز از پیش‌بینی fixed-horizon return به outcome وابسته به مسیر قیمت منتقل می‌شود. این انتخاب از نظر هزینه و provenance کم‌ریسک‌تر از خرید دادهٔ On-Chain است.

### ۳.۱. مانع تکرار آزمایش قبلی

با این حال، Triple-Barrier + Logistic Regression دقیقاً قبلاً در ADR-056 اجرا شده است: ورود open(t+1)، موانع متقارن ±2×ATR(14)، time barrier برابر ۲۴ کندل، برچسب upper-first/lower-first و حذف TIMEOUT/AMBIGUOUS/UNAVAILABLE از fit. نتیجهٔ pooled OOF accuracy=51.09% در برابر شرط سخت >55% بود و مسیر بسته شد. اجرای مجدد همین تعریف، با همان data hash و همان ۱۱ feature، فقط تکرار نتیجهٔ ردشده است و توصیه نمی‌شود.

### ۳.۲. تنها شکل قابل‌بررسی برای A، یک فرضیهٔ متمایز است

• Vertical timeout باید به‌عنوان outcome تعریف‌شده وارد target شود، نه اینکه همهٔ timeoutها بدون ارزیابی از کلاس دودویی حذف شوند. دو طراحی قابل پیش‌ثبت: سه‌کلاسهٔ TP_FIRST / SL_FIRST / VERTICAL_TIMEOUT؛ یا برچسب سود/زیان خالص که خروج عمودی را نیز در نتیجه لحاظ کند.

• Barrier multipliers، time horizon و policy رخدادهایی که در یک OHLC bar هم TP و هم SL را لمس می‌کنند، باید قبل از هر اجرا freeze شوند. AMBIGUOUS نباید به‌دلخواه به TP یا SL تبدیل شود.

• Entry timing همان close(t) decision و open(t+1) entry reference؛ featureها فقط از کندل‌های بسته‌شده. Purge بر اساس کل label window و embargo زمانی اجباری است.

• هزینهٔ round-trip در label/economic ledger لحاظ شود؛ در نهایت NetPF، trade count، MaxDD و fold stability در Cost-50 و Cost-90 بررسی شوند. Accuracy معیار تشخیصی است، نه پذیرش اقتصادی.

• Volume در دیتاست PRE-HACK بررسی‌شده صفر است؛ بنابراین predictor حجمی از همین فایل سیگنال ندارد. ویژگی‌های قیمت/زمان باید تنها predictorهای قابل‌دفاع باشند مگر اینکه مالک بعداً مجوز دادهٔ دیگری بدهد.

محدودیت: PRE-HACK قبلاً در معرض پژوهش بوده و فقط exploratory است؛ حتی نتیجهٔ مثبت آن confirmatory محسوب نمی‌شود و Holdout را باز نمی‌کند.

## ۴. گزینهٔ B — تغییر پارادایم داده به On-Chain

ADR-061 امکان‌سنجی منابع را انجام داده است؛ بررسی قیمت‌نامهٔ رسمی CryptoQuant در ۲۷ نوامبر ۲۰۲۶ نیز انجام شد. On-chain می‌تواند صرفاً متغیر زمینه‌ای باشد؛ حذف کامل قیمت/بازده BTCUSDT، سنجش سودآوری و اجرای قابل‌ممیزی را بی‌تعریف می‌کند. این فاز هنوز data acquisition، purchase یا model authorization نیست.

منبع / طرح

قیمت/entitlement رسمی بررسی‌شده در 27-11-2026

محدودیت اصلی

CryptoQuant Basic / Advanced

Basic رایگان: 10k credits/mo، 10 req/min، 30-day history. Advanced: $39/mo یا $29/mo سالانه؛ 300k credits، 60 req/min، یک‌سال history؛ هر دو فقط Market API.

هیچ‌کدام On-Chain API ندارند؛ Advanced trial هفت‌روزه با کارت پرداختی خودکار تبدیل به paid می‌شود.

CryptoQuant Professional

اولین API tier دارای On-Chain: $109/mo یا $99/mo با billing سالانه ($1,188/year)؛ 500k credits/mo، 120 req/min، یک‌سال API history.

License در قیمت‌نامه Personal Use است؛ برای استفادهٔ تجاری/Business کافی تلقی نشود. FAQ: On-Chain روزانه؛ Nobitex entity تأیید نشده.

CryptoQuant Premium

$799/mo فقط با billing سالانه؛ حدود $9,588/year؛ unlimited credits، 800 req/min، full history و on-chain hourly/per-block.

تنها tier دارای Business Use license در ماتریس رسمی؛ گران و همچنان Nobitex entity/metric coverage اثبات‌نشده.

Glassnode Professional

Professional از $999/mo شروع می‌شود؛ API add-on اختیاری است و Data Credits جداگانه/پیکربندی‌شده‌اند؛ هزینهٔ نهایی API در صفحهٔ عمومی ثابت نیست.

API در Advanced نیست؛ Professional شامل PIT و Commercial Use در ماتریس قیمت‌نامه است؛ Nobitex per-exchange coverage و هزینهٔ نهایی باید کتبی تأیید شود.

Coin Metrics Community

رایگان برای Community endpointها؛ مجوز ثبت‌شده non-commercial.

Metric entitlement دقیق exchange net-flow تأیید نشده و استفادهٔ تجاری با مجوز Community سازگار نیست.

خطر روش‌شناختی: BTC inflow/outflow به صرافی لزوماً به معنی فروش در Nobitex BTCUSDT نیست. اگر historical wallet labels پس از واقعیت اصلاح شوند و archive نقطه‌به‌نقطه نگهداری نشود، backtest دچار revision leakage می‌شود. Alignment باید بر اساس available_at واقعی باشد؛ daily observation تکرارشده روی کندل‌های 1h/4h نمونهٔ مستقل جدید تولید نمی‌کند.

## ۵. توصیه و وضعیت تصمیم

توصیهٔ مشروط: از نظر هزینه و دادهٔ موجود، A نسبت به B ارجح است، اما نه به‌شکل تکرار ADR-056. فقط یک target redesign که vertical timeout را به‌صورت صریح مدل کند و economics-after-cost را در هدف/ledger منعکس کند ارزش تدوین preregistration جدید دارد. B فعلاً DEFER/HOLD است: پیش از هر هزینه یا دریافت داده، Nobitex entity coverage، metric entitlement، PIT availability، retention، license و هزینهٔ کل باید مستند شوند.

هیچ‌یک از گزینه‌ها انتخاب نشده است. وضعیت = OWNER DECISION PENDING. این نسخه هیچ مدل، کد، data download، subscription، fit، backtest، Holdout access یا Paper/Live را مجاز نمی‌کند. گام بعدی فقط پس از انتخاب مالک و با preregistration مستقل تعیین می‌شود.

Safety invariants: Champion=null | order_permission=false | trading_authorization=NO-GO | Holdout=SEALED | posthack_data_read=false | private_api_calls=0 | real_order_attempts=0.

## ۰. کنترل سند، مرجعیت و نحوهٔ استفاده

مشخصه

مقدار

عنوان

AliX — سند مرجع جامع معماری، حاکمیت و نقشهٔ راه

نسخه

۱٫۳؛ افزودن M6 و M7 به Trial Registry، ثبت M7=CLOSED_NO_GO، M8=BLOCKED و آغاز M8.1 Capture داده BTCUSDT؛ شواهد ADR-050 تا ADR-075

تاریخ بازبینی

۲۷ نوامبر ۲۰۲۶

شاخهٔ پژوهشی فعال

research/nobitex-hf-strategy

HEAD پایهٔ تأییدشده

95133203e70265ac96f7d4dd7c14ed213fadaf38 (HEAD پایهٔ مشاهده‌شده پیش از commit مستندات v1.2)

شاخهٔ main

a1c1a658906cd79be62796885e3ed782b66a267d — main HEAD فعلی؛ Tree SHA=5761c6c34143f8e6924f53a4e745c48932a5f791 (در این به‌روزرسانی تغییر نکرده)

آخرین Standard CI روی HEAD پایه

Standard CI #46001067175 | Run ID 46001067175 | completed / success | Coverage 90.36%

وضعیت R3

HGBQ-CQR / OFI / Sentiment = CLOSED NO-GO؛ Regime T Triple-Barrier قبلی نیز NO-GO. گزینهٔ فاز بعدی هنوز انتخاب نشده است.

دامنه

Nobitex Spot/USDT؛ شواهد اقتصادی مسیرهای بسته‌شده؛ پیشنهاد طراحی A/B؛ بدون مدل‌سازی، Fit یا دادهٔ جدید در این نسخه.

آخرین Four-Arm Quality Preflight

Four-Arm Quality Preflight #45998903704 روی SHA 95133203… = SUCCESS؛ Standard CI #46001067175 روی همان SHA = SUCCESS.

### ۰.۱. قاعدهٔ تقدم منابع

### 1. Owner Directive صریح و جدیدتر، در محدودهٔ موضوع خود بر سند تاریخی مقدم است؛ اما به‌تنهایی پیاده‌سازی runtime یا مجوز اجرا ایجاد نمی‌کند.

#### 3. این سند وضعیت شاخه و CI را فقط در SHA و Run IDهای صریحاً درج‌شده به‌عنوان مشاهدهٔ مستقیم ثبت می‌کند. وضعیت تاریخی v1.1 و مقدارهای قبلی HEAD، وضعیت زنده تلقی نمی‌شوند. بخش v1.2 در ابتدای سند مرجع جاری است و بر بخش‌های تاریخی v1.1 تقدم دارد.

#### 3. این سند وضعیت شاخه و CI را فقط در SHA و Run IDهای صریحاً درج‌شده به‌عنوان مشاهدهٔ مستقیم ثبت می‌کند. مقادیر تاریخی نسخه‌های قبلی بدون بازبینی، وضعیت زنده تلقی نمی‌شوند.

### 4. هیچ متن این سند Holdout را باز نمی‌کند، مجوز معامله صادر نمی‌کند یا کیفیت اقتصادی مدل را از روی CI نتیجه نمی‌گیرد.

### ۰.۲. تغییرات عمده نسبت به نسخهٔ ۰٫۹

حوزه

وضعیت نسخهٔ ۰٫۹

به‌روزرسانی نسخهٔ ۱٫۱

جهت پژوهش

Regime R هنوز به انتخاب نهایی مدل نیاز داشت.

ADR-047/048 مسیر اولیه را ثبت کردند؛ ADR-049 مسیر فعال را به HGBQ-CQR تغییر می‌دهد.

زیرساخت R1/R2

در نسخهٔ قبلی ثبت نشده بود.

۳۰ Trial ID، قرارداد target/feature/distribution و تست‌های Mock-only موجودند.

CI شاخهٔ فعال

CI قرمز گزارش شده بود.

CI #2450 روی SHA فعلی کاملاً success شده است.

Complexity

رفع Xenon هنوز قطعی نبود.

تابع validate_distribution_forecast refactor و Xenon PASS شده است.

Dependency

NGBoost/MAPIE بررسی نشده بودند.

NGBoost 0.6.0 و نسخه‌های بررسی‌شده فاقد PyPI provenance؛ MAPIE 1.6.0 دارای provenance کامل؛ scikit-learn 1.9.0 از قبل pin و lock است؛ نصب MAPIE هنوز انجام نشده.

Roadmap

مسیر NGBoost + CQR و عبور از dependency/API validation.

HGBQ-CQR به‌عنوان fallback انتخاب شده؛ G1 تا افزودن MAPIE به lock و عبور از resolution/audit همچنان BLOCKED است؛ سپس API و preregistration.

Standard CI #2450 فقط سلامت مهندسی، قراردادها و تست‌های موجود در SHA 3162f9a… را تأیید می‌کند. دو job پژوهش real-data در آن skipped بودند. Four-Arm Quality Preflight #46 موفق بود، اما روی SHA e842187… اجرا شد و شاهد Preflight روی HEAD فعلی نیست. هیچ Training، Fit، Inference، Backtest یا نتیجهٔ اقتصادی از این شواهد حاصل نشده است.


راهنمای ساختار سند

بخش

موضوع

۱

خلاصهٔ مدیریتی و وضعیت کنونی

۲

نقشهٔ مخزن، Commitها و شواهد CI

۳

حاکمیت تصمیمات و ADRها

۴

شواهد تاریخی اقتصادی و درس‌آموخته‌ها

۵

معماری کلان و مرز مسئولیت اجزا

۶

قرارداد پژوهشی Regime R: زمان‌بندی، target و feature

۷

HGBQ-CQR، MAPIE ConformalizedQuantileRegressor و منطق Lower-Bound Gate

۸

Trial Registry، ۳۰ شناسه و معیارهای پذیرش

۹

حاکمیت داده، Holdout و استقلال آماری

۱۰

Dependency، provenance و سازگاری نسخه‌ها

۱۱

CI، کیفیت مهندسی و Change Control

۱۲

ریسک، امنیت و Invariantهای غیرقابل‌مذاکره

۱۳

نقشهٔ راه جدید و Gateهای مرحله‌ای

۱۴

شرایط ورود به R3 و معیارهای پذیرش

۱۵

پروتکل ادامهٔ کار و انتقال به مهندس بعدی

۱۶

واژه‌نامه و منابع

پیوست A

۳۰ Trial ID پیش‌ثبت‌شده

پیوست B

تاریخچهٔ Commitها و Runهای CI

### ۱. خلاصهٔ مدیریتی و وضعیت کنونی در snapshot تاریخی v1.2

AliX یک سامانهٔ پژوهش و تصمیم‌سازی کنترل‌شده برای Nobitex Spot/USDT است. تا ۲۷ نوامبر ۲۰۲۶، مسیرهای اقتصادی HGBQ-CQR، M3 OFI و M5.2 Sentiment با حکم NO-GO بسته شده‌اند؛ آزمایش مستقل Regime T با Target سه‌گانهٔ Triple-Barrier نیز قبلاً NO-GO شده است. هیچ استراتژی دارای لبهٔ اقتصادی قابل دفاع پس از هزینه برای Paper/Live اثبات نشده است. گزینه‌های بعدی در v1.2 صرفاً پیشنهاد طراحی‌اند و انتخاب مالک را می‌طلبند.

موضوع

وضعیت

شاهد / توضیح

HGBQ-CQR / Regime P

CLOSED — NO-GO

ADR-053 تا ADR-055؛ سه horizon candidate یکتا؛ ۳۰ slot شامل ۲۷ duplicate؛ هیچ Lower Bound از Cost-50/90 عبور نکرد؛ ۰ معامله.

M3 OFI

CLOSED_NO_GO

ADR-059/060؛ دو Trial منطقی 1h/4h؛ هر دو رد شدند؛ مجموعهٔ دادهٔ تاریخی BTCIRT است و به BTCUSDT تعمیم داده نمی‌شود.

Regime T Triple-Barrier

CLOSED — NO-GO

ADR-056؛ یک آزمایش Logistic؛ pooled OOF accuracy=51.09% در برابر آستانهٔ سخت >55%؛ balanced accuracy=51.57%.

M5.2 Sentiment

CLOSED — NO-GO

ADR-062 تا ADR-064؛ چهار Trial Control/Sentiment در 1h/4h؛ Sentiment در چهار مقایسه NetPF و PnL را بدتر کرد.

Standard CI

PASS

Run #46001067175 روی SHA 95133203…؛ 1,127 passed، 5 skipped، coverage=90.36%.

پیشنهاد مسیر بعدی

OWNER DECISION PENDING

A فقط با Target واقعاً متفاوت از Regime T قبلی؛ B مشروط به حل هزینه، پوشش Nobitex و point-in-time provenance.

GLOBAL_N / DSR

UNKNOWN / NOT COMPUTABLE

هیچ‌یک از این مسیرها multiplicity تاریخی کامل را حل نکرده‌اند؛ DSR عددی ساخته نمی‌شود.

Holdout

SEALED

هیچ بایت Holdout یا Post-Hack در آزمایش‌های ثبت‌شدهٔ این فاز خوانده نشده است.

Paper / Live

NO-GO

هیچ Champion یا مجوز سفارش صادر نشده است.

ایمنی

FAIL-CLOSED

Champion=null؛ order_permission=false؛ trading_authorization=NO-GO؛ private_api_calls=0؛ real_order_attempts=0.

### ۱.۱. چه چیزی انجام شده و چه چیزی نشده است؟

انجام‌شده: قراردادهای پایه و Mock-only، Trial Registry تاریخی، ارزیابی HGBQ-CQR و ablation، ارزیابی دو Trial OFI، آزمایش Triple-Barrier/Logistic و ارزیابی چهار Trial M5.2؛ همهٔ مسیرهای اقتصادی فوق با حکم NO-GO بسته شده‌اند. ADR-064 نتیجهٔ نهایی M5.2 را ثبت می‌کند.

انجام‌نشده و مجازنشده: انتخاب یا پیش‌ثبت نسخهٔ بعدی Target، ساخت مدل جدید، Fit، tuning، inference، backtest یا دریافت دادهٔ تازه. هیچ انتخابی میان گزینه‌های A و B انجام نشده و هیچ مجوزی برای اجرای آن‌ها صادر نشده است.

تصمیم جاری: HGBQ-CQR به‌علت شکست Lower-Bound Gate و COMPLETE_ABSTENTION_REPEATED بسته است؛ OFI به‌علت NO-GO / INSUFFICIENT_SAMPLE و تصمیم صریح مالک بسته است؛ Sentiment به‌علت بدترشدن NetPF و PnL نسبت به Control بسته است. Triple-Barrier قبلی نیز با accuracy=51.09% در برابر آستانهٔ >55% رد شد. هر مسیر تازه نیازمند فرضیهٔ واقعاً جدید، preregistration مستقل و مجوز جداگانه است.

Regime R از نظر جهت‌گیری و قراردادهای Mock-only آمادهٔ ادامهٔ طراحی است؛ مسیر مدل فعال HGBQ-CQR است. بااین‌حال R3 برای پیاده‌سازی مدل یا Fit آماده نیست. باز شدن R3 مستلزم تکمیل G1 با pin/lock/audit MAPIE، عبور G2 از API compatibility، تکمیل و Hash کردن preregistration، مجوز مستقل داده/fit و مجوز مستقل مالک است.


## ۲. نقشهٔ مخزن، Commitها و شواهد CI

### ۲.۱. شاخه‌ها

شاخه

HEAD / وضعیت

سیاست

research/nobitex-hf-strategy

95133203e70265ac96f7d4dd7c14ed213fadaf38

شاخهٔ فعال پژوهش؛ این SHA پایهٔ مستندات v1.2 است. commit خود این سند یک commit مستنداتی جدید و پس از آن خواهد بود.

main

a1c1a658906cd79be62796885e3ed782b66a267d / Tree 5761c6c34143f8e6924f53a4e745c48932a5f791

Protected؛ در این نوبت تغییر داده نشده است. Tree SHA دقیقاً برابر مقدار baseline ثبت‌شده است.

release/nobitex-mvp

در v0.9 paper-only ثبت شده بود

اجرای زنده مسدود؛ وضعیت HEAD در این نوبت دوباره ممیزی نشده است.

staging

در v0.9 حاوی strategy_selection ثبت شده بود

ادغام یا promotion آن در این نوبت انجام نشده است.

### ۲.۲. زنجیرهٔ اصلاح CI در شاخهٔ فعال و ممیزی Preflight

Commit

پیام / محدوده

CI مرتبط و نتیجه

e842187d1491ba50b925c4d3d300fdab6f441b7c

fix(regime_r): resolve CI format and test issues؛ فقط test_regime_r_ngboost_cqr_contracts.py

CI #2448 / 43833782132: FAILURE؛ Xenon Rank C در contracts.py.

d98df5ea57e28e1420668983afba586874ed5d6f

refactor(regime_r): reduce complexity of validate_distribution_forecast؛ فقط contracts.py؛ استخراج helperهای خصوصی

CI #2449 / 43836687806: FAILURE؛ Ruff Format یک blank line دیگر می‌خواست.

3162f9a88efb81b7069f63afe354738df689a869

style(regime_r): add missing blank line between top-level functions؛ دقیقاً ۱ addition و ۰ deletion در contracts.py

CI #2450 / 43860221606: SUCCESS کامل.

### ۲.۳. فایل‌های R1/R2 در HEAD تأییدشده

مسیر

وضعیت در HEAD فعلی

کارکرد

research/regime_r/ngboost_cqr/contracts.py

موجود؛ نام مسیر میراثی است

TargetWindow، FeatureSnapshot، DistributionForecast، ConformalInterval، validatorها و cost gate؛ هنوز مدل HGBQ-CQR یا adapter در runtime پیاده نشده است.

research/regime_r/ngboost_cqr/__init__.py

موجود

Namespace بستهٔ Regime R.

tests/unit/test_regime_r_ngboost_cqr_contracts.py

موجود

تست‌های قرارداد Mock-only؛ ۳۰ شناسه، زمان‌بندی، causal features، quantileها، CQR و cost gate.

research/regime_r/ngboost_cqr/adapters.py

وجود ندارد

Adapter/interface واقعی HGBQ-CQR ساخته نشده است؛ ساخت آن خارج از این نوبت و مشروط به G1/G2 و مجوز است.

requirements.in

وجود ندارد

CI در حال حاضر از requirements.txt به‌عنوان ورودی pip-compile استفاده می‌کند.

requirements.txt / requirements.lock

موجود؛ MAPIE و NGBoost هر دو غایب‌اند

scikit-learn==1.9.0 موجود است؛ MAPIE 1.6.0 فقط گزینهٔ انتخاب‌شده برای نصب آینده است و هنوز lock نشده.

تفسیر Run #2450

Quality، PostgreSQL Integration، Freqtrade Runtime Smoke و Four-Arm Lock Bare Environment همگی success شدند. Jobهای ensemble-real-data-research و real-data-diagnostic skipped بودند. نتیجهٔ سبز CI فقط به همین commit و scope اجراشده مربوط است؛ نه به اقتصاد مدل یا آمادگی معامله.

## ۳. حاکمیت تصمیمات و دفتر ADR

ADRهای قدیمی همچنان سابقهٔ تصمیم را نگه می‌دارند. ADR-047 و ADR-048 تصمیم‌های جدید را اضافه می‌کنند، نه اینکه نتایج تاریخی، تعداد trialهای نامعلوم یا مرزهای Holdout را بازنویسی کنند.

مرجع

تصمیم اصلی

وضعیت / اثر جاری

ADR-012

پیش‌ثبت، Trial Registry و Holdout Integrity.

حاکم بر ثبت همهٔ candidateها و منع مصرف Holdout.

ADR-036

ریسک دوام/منشأ داده و مرز دسترسی.

حاکم بر دادهٔ جدید و post-hack.

ADR-039

وضعیت دوگانهٔ seal/lock برای Holdout Nobitex.

Holdout همچنان sealed؛ این فاز آن را باز نمی‌کند.

ADR-045

حسابداری trial و امکان‌پذیری DSR.

GLOBAL_N=UNKNOWN؛ run/attempt/candidate از هم تفکیک شوند.

ADR-046

منشأ مدل، استقلال داده و مرز نشت.

pretraining/overlap و دادهٔ تاریخی باید صریح گزارش شوند.

ADR-047

احیای Regime R و pivot به خانواده‌های مدل جدید.

پذیرفته برای جهت‌گیری؛ R1/R2/R3 تحت مجوزهای جدا.

ADR-048

انتخاب NGBoost + CQR برای Regime R.

Owner-selected; R1/R2 contracts + mocks approved; empirical execution NOT AUTHORIZED.

ADR-049

رد NGBoost به‌علت فقدان provenance و انتخاب HGBQ-CQR + MAPIE 1.6.0؛ حفظ ۳۰ slot با parent/supersession lineage.

تصمیم مسیر وابستگی و قرارداد؛ G1 هنوز تا lock/audit و G2 تا API verification بسته نمی‌شود؛ هیچ اجرای تجربی مجاز نیست.

ADR-050

HGBQ-CQR adapter/API compatibility

تاریخی؛ پس از اجرای G4/G5 و ablation نهایی، مسیر اقتصادی طبق ADR-055 بسته شد.

ADR-051

HGBQ-CQR preregistration commitment

پیش‌ثبت تاریخی؛ تغییرات پس از مشاهده ممنوع؛ مسیر اقتصادی NO-GO.

ADR-052

Regime R execution gate / config enforcement

قرارداد fail-closed؛ جایگزین مجوز اقتصادی نیست.

ADR-053

HGBQ-CQR G4/G5 evaluation

NO-GO / INSUFFICIENT_EVIDENCE؛ ۳ horizon candidate یکتا؛ صفر Lower Bound بالاتر از هزینه.

ADR-054

HGBQ-CQR observability / abstention diagnosis

Diagnosis complete؛ وضعیت اقتصادی همچنان NO-GO.

ADR-055

HGBQ-CQR G4.6 volume ablation

COMPLETE_ABSTENTION_REPEATED؛ مسیر اقتصادی HGBQ-CQR طبق تصمیم مالک بسته شد.

ADR-056

Regime T Triple-Barrier + Logistic

CLOSED — NO-GO؛ pooled OOF accuracy=51.09% و gate مالک >55%.

ADR-057

Market microstructure feasibility

فقط feasibility؛ L2/tick عمومی در زمان واقعی مستند، آرشیو تاریخی کافی اثبات نشده.

ADR-058

Historical sufficiency / live capture

M3 data capture historical؛ محدودیت پیوستگی/داده و مرز BTCIRT ثبت شده است.

ADR-059

M3 OFI preregistration

دو Trial منطقی 1h/4h؛ پیش‌ثبت قفل‌شده، اجرای بعدی با مجوز جدا.

ADR-060

M3 OFI baseline evaluation

CLOSED_NO_GO؛ هر دو Trial رد شدند؛ 1h NetPF=0.07347؛ 4h NetPF=0.23269؛ منبع BTCIRT.

ADR-061

BTCUSDT on-chain/sentiment feasibility

Feasibility only؛ CryptoQuant Pro در snapshot برابر $109/mo؛ پوشش Nobitex و PIT unresolved؛ هیچ خرید/دریافت داده مجاز نشد.

ADR-062

M5 sentiment capture/preregistration

چهار Trial منطقی ثبت شدند؛ capture snapshot موجود و پیش‌ثبت قفل شد؛ NO-GO execution pending در آن مرحله.

ADR-063

M5.2 dispatch boundary

مانع workflow_dispatch و راهکار موقت default-branch را ثبت کرد؛ توسط ADR-064 supersede شد.

ADR-064

M5.2 BTCUSDT sentiment evaluation

ECONOMIC NO-GO؛ چهار Trial؛ هر چهار paired comparison علیه Sentiment؛ 16 fold fits.

ADR-065

MRD v1.2 / target redesign decision

این سند؛ وضعیت جاری، Trial inventory و مقایسه گزینه‌های A/B؛ انتخاب مسیر بعدی به تصمیم مالک موکول است.

### ۳.۱. جایگاه ADR-047

ADR-047 تشخیص می‌دهد که شکست‌های قبلی را نمی‌توان فقط با تعویض الگوریتم توضیح داد. Teacherهای OHLCV پیشین ضعیف بودند؛ PPO در نقش allocator هیچ سیگنالی را نپذیرفت و این رفتار اثبات موفقیت ریسک یا شکست مستقل پیش‌بینی جهت نیست. Run #10 نیز ۰ PASS از ۱۲ سلول و شکست پایداری fold گزارش کرده بود. چهار مسیر OFI/DeepLOB، NGBoost+CQR، BOCPD و PatchTST در ADR-047 اولویت‌بندی شدند؛ این فهرست گزینه‌های طراحی بود، نه مجوز اجرا.

### ۳.۲. جایگاه ADR-048

ADR-048 انتخاب تاریخی مالک را به NGBoost + CQR محدود کرد. ADR-049 این مسیر را برای اجرای آینده، در محدودهٔ dependency/provenance و learner پایه، supersede می‌کند: NGBoost به‌علت نبود provenance رد و HGBQ-CQR انتخاب می‌شود. پرسش پژوهشی همچنان این است که آیا lower prediction bound کالیبره‌شده پس از هزینه‌ها و در چند fold زمانی پایدار می‌ماند.

قید حاکمیتی ADR-048

ADR-048 هیچ real-data read، fit، tuning، inference، backtest، Holdout access، dispatch، nonce، private API یا order را مجاز نکرده است. انتخاب خانواده فقط مجوز طراحی قرارداد و تست Mock-only است.

## ۴. شواهد تاریخی اقتصادی و درس‌آموخته‌ها

نتایج تاریخی نسخه‌های ۰٫۸ و ۰٫۹ حفظ می‌شوند. این جدول برای جلوگیری از بازاجرای بی‌دلیل خانواده‌های قبلی و جلوگیری از بازنویسی نتایج است؛ اعداد مربوط به مسیرهای متفاوت، قرارداد هزینه و مجموعه‌دادهٔ متفاوت‌اند و نباید با یکدیگر ادغام شوند.

مسیر

شواهد منتخب ثبت‌شده

حکم و پیامد

AI-6 / PPO

Validation baseline: NetPF 0.3743، MaxDD 66.56%، 234 معامله؛ terminal baseline: NetPF 0.4370، MaxDD 60.74%، 225 معامله. PPO: صفر پذیرش از 228 و 231 سیگنال.

CLOSED — NO STATISTICAL EDGE FOUND IN RL؛ PPO مدل پیش‌بینی جهت نبود.

Run #10 / Chronos-Kronos

Best observed NetPF: 0.6324 در 50 bps و 0.3936 در 90 bps؛ 36 بررسی پایداری fold ناموفق.

NO EDGE؛ DSR کلی قابل محاسبه نیست؛ trial accounting تاریخی ناقص.

TTM-R2

QLIKE از 0.39347957 به 0.27785937 بهبود یافت؛ replay معاملاتی NetPF=0.3748 و MaxDD=18.26% گزارش شد.

پیش‌بینی نوسان به‌عنوان جزء ریسک؛ نه سیگنال ورود مستقل.

DEAP AI-2 / OHLCV

OOS بسیار کم‌معامله؛ ۴ معامله در گزارش نسخهٔ ۰٫۹؛ نسخه‌های قبلی نیز نتایج ناپایدار داشتند.

NO-GO؛ خانواده تکرار نمی‌شود مگر فرضیه و دادهٔ جدید.

Meta-Labeling

در سه fold مصنوعی، baseline از نظر Brier/log-loss بهتر از فیلتر شد؛ بازده میانگین با فیلتر بدتر شد.

مکانیک تأیید؛ ارزش اقتصادی اثبات نشد.

Chronos-2 Ensemble

پیاده‌سازی قبلی گزارش‌شده؛ حکم اقتصادی رسمی/تأییدی برای آن منتشر نشده بود.

نامعین؛ عدم استفاده به‌عنوان شاهد edge.

HGBQ-CQR / Regime P

G4: 3 candidate یکتا در افق‌های 1h/4h/24h؛ ۳۰ slot شامل ۲۷ duplicate؛ 90 fit؛ G4.6 volume ablation نیز 0/3 candidate pass و 90/90 diagnostic برابر G4.5.

CLOSED — NO-GO؛ هیچ forecastای از هزینه عبور نکرد؛ zero hypothetical trades.

M3 OFI / Regime M

۲ Trial منطقی (1h و 4h) روی پنج capture segment از BTCIRT؛ 1h: 665 trade / NetPF 0.07347؛ 4h: 82 trade / NetPF 0.23269.

CLOSED_NO_GO با تصمیم صریح مالک؛ 0/2 Trial پذیرفته.

Regime T / Triple-Barrier

31,801 input rows؛ 24,740 binary events؛ 7,061 timeout/ambiguous/unavailable excluded؛ 1,652 pooled OOF prediction. Accuracy=51.09%.

CLOSED — NO-GO؛ آستانهٔ preregistered >55% برآورده نشد؛ این یک economic backtest نبود.

M5.2 / Sentiment

۴ Trial منطقی؛ ۱۶ fold fits؛ 101,648 OOS prediction rows؛ دو horizon و دو cost scenario.

CLOSED — NO-GO؛ Sentiment در هر چهار paired comparison از Control بدتر بود.

### ۴.۱. اصول تفسیر نتایج

صفر معامله یا Drawdown صفر ناشی از عدم مواجهه، اثبات مدیریت ریسک موفق نیست.

Forecast accuracy، QLIKE، loss یا Fitness بالاتر به‌تنهایی اثبات سود خالص پس از هزینه نیست.

بهبود کیفیت پیش‌بینی باید به ledger اجرایی هزینه‌شده، تعداد معاملهٔ کافی، drawdown و پایداری زمانی متصل شود.

نتیجهٔ NO-GO معتبر است و نباید برای رسیدن به GO، آستانه‌ها، trial budget یا کیفیت Gateها پس از مشاهدهٔ نتایج تغییر کنند.

## ۵. معماری کلان و مرز مسئولیت اجزا

### ۵.۱. معماری دوصفحه‌ای

RESEARCH PLANE (offline / read-only / no secrets)
Market Artifact + Manifest
  -> Data Quality Gate
  -> Causal Feature Builder
  -> Regime R Forecast Research (HGB quantile learners + MAPIE CQR)
  -> Chronological Evaluator
  -> Trial Registry / Research Ledger / Artifact Archive

DECISION / EXECUTION PLANE (separate authorization boundary)
Versioned DataProvider
  -> Data Quality Gate
  -> FreqAI / Frozen Artifact
  -> Decision Proposal
  -> Independent Risk Gate / Supervisor
  -> Freqtrade Order Lifecycle
  -> Nobitex Exchange Adapter

R1/R2 در Research Plane قرار دارند. خروجی احتمالاتی مدل در صورت تکمیل آینده می‌تواند یک proposal بسازد؛ اما اختیار سفارش یا تغییر policy ندارد. هر artifact باید نسخه، hash، schema، data lineage و مجوز مصرف مستقل داشته باشد.

### ۵.۲. مرز مسئولیت اجزا

جزء

مسئولیت

محدودیت غیرقابل‌مذاکره

Freqtrade

مالک انحصاری lifecycle سفارش، fill/cancel/status، sync و trade DB.

ساخت executor موازی یا lifecycle پنهانی ممنوع.

AliXStrategy

تبدیل artifact مجاز به Decision Proposal.

REST order مستقیم یا دورزدن Risk Gate ممنوع.

Regime R / HGBQ-CQR

پیش‌بینی quantileهای lower/median/upper با HGB و ساخت بازهٔ conformal با MAPIE CQR؛ abstention.

تولید توزیع کامل احتمالاتی، PDF/CDF یا پارامترهای mean/dispersion بدون مدل/برآورد مستقل مجاز نیست؛ اختیار سفارش ندارد.

Research Plane

آزمایش آفلاین، Trial Registry و گزارش.

هیچ دسترسی private API/order/nonce ندارد.

Data Quality Gate

schema، freshness، gap، provenance و point-in-time.

gap یا feature نامعتبر را خاموش/پنهان نمی‌کند.

Risk Gate/Supervisor

رأی مستقل allow/reject/resize و Kill Switch.

از مدل فرمان نمی‌گیرد؛ reset خودکار ممنوع.

Decision Ledger

ثبت append-only تصمیم، دلیل، revision و نتیجه.

جایگزین Freqtrade DB یا exchange truth نیست.

### ۵.۳. Freqtrade و Nobitex

مسیر مورد انتظار سفارش، صرفاً در آینده و پس از Release Authorization، برابر Freqtrade → Adapter → Nobitex است. نسخهٔ release/nobitex-mvp در MRD v0.9 paper-only و LiveExecutionBlocked ثبت شده بود. مسیرهای post-only، کلید امضا، cancel/replace، idempotency، unknown-order reconciliation، partial fill و rate limit همچنان باید در سطح adapter و lifecycle اثبات شوند. این سند مجوز پیاده‌سازی یا استفادهٔ live صادر نمی‌کند.

## ۶. سوابق قرارداد پژوهشی Regime R — مسیر تاریخی HGBQ-CQR (بسته‌شده)

### ۶.۱. قرارداد تصمیم و هدف

واحد پایه داده، کندل کامل یک‌ساعته است. در پایان کندل t، ویژگی‌ها فقط با داده‌ای محاسبه می‌شوند که تا close همان کندل در دسترس بوده است. ورود مرجع، open کندل t+1 است و خروج مرجع برای افق h، close کندل t+h است.

decision_time = close(t)
entry_reference = open(t + 1)
exit_reference(h) = close(t + h)
h in {1, 4, 24}
gross_return(t,h) = close[t+h] / open[t+1] - 1
label_end_index = t + h

تمایز حیاتی: افق ۲۴ ساعته با تایم‌فریم ورود روزانه یکی نیست

h=24 یعنی بازده هدف طی ۲۴ کندل یک‌ساعته؛ این به معنی استفاده از کندل 1d، cadence ورود روزانه یا مجازشدن 1d در Opportunity Hunter نیست. Allowlist ورود قبلی 15m/30m/1h/4h حفظ می‌شود؛ 1d فقط با دستور مستقل و صرفاً برای آزمایش آفلاین تشخیصی قابل بررسی است و 1w ممنوع باقی می‌ماند.

### ۶.۲. افق‌ها و Trial Budget

افق پیش‌بینی

مبنای محاسبه

تعداد slot

1h

یک کندل ۱ساعته تا close[t+1]

۱۰

4h

چهار کندل ۱ساعته تا close[t+4]

۱۰

24h

۲۴ کندل ۱ساعته تا close[t+24]

۱۰

کل

یک خانوادهٔ selection شامل هر سه افق

۳۰

### ۶.۳. Feature Schema اولیه

نام Feature
تعریف سطح قرارداد

log_return_1

بازده لگاریتمی یک کندل

log_return_4

بازده لگاریتمی چهار کندل

log_return_24

بازده لگاریتمی ۲۴ کندل

realized_volatility_6

نوسان تحقق‌یافتهٔ trailing ۶ کندل

realized_volatility_24

نوسان تحقق‌یافتهٔ trailing ۲۴ کندل

range_fraction_1

(High-Low) نرمال‌شده با قیمت مرجع

atr_fraction_14

ATR(14) نرمال‌شده با Close

log_volume_change_1

تغییر لگاریتمی حجم، فقط گذشته

volume_zscore_24

z-score حجم در پنجرهٔ trailing ۲۴

hour_sin / hour_cos

کدگذاری چرخه‌ای ساعت تصمیم

weekday_sin / weekday_cos

کدگذاری چرخه‌ای روز هفته

فرمول دقیق هر feature، warm-up، timezone، رفتار volume=0، normalization و serialization باید در schema نسخه‌دار ثبت و hash شود. هر تغییر در نام، فرمول، پنجره یا normalization یک candidate جدید است. ویژگی‌ها right-aligned و مبتنی بر کندل بسته‌شده‌اند؛ backfill، centered window، scaler با دادهٔ آینده، incomplete candle و feature مشتق از label ممنوع‌اند.

## ۷. سوابق فنی HGBQ-CQR و MAPIE — فقط برای lineage تاریخی؛ مسیر اقتصادی CLOSED / NO-GO

☐ برای B: Nobitex entity coverage، point-in-time history، vendor licensing، cost/credit budget و provenance باید پیش از هر خرید یا data access اثبات شوند.

HistGradientBoostingRegressor با loss="quantile" برای برآورد مستقیم quantileهای هدف به‌کار می‌رود. برای سطوح lower، median=0.5 و upper، مدل quantile متناظر آموزش داده می‌شود. HGBQ-CQR یک برآوردگر توزیع کامل (distributional density model) نیست؛ خروجی معتبر آن quantile predictions است. پارامتر regularization در sklearn با نام l2_regularization شناخته می‌شود؛ مقدار آن باید در preregistration تعیین و freeze شود؛ این سند مقدار عددی پیش‌فرضی را تثبیت نمی‌کند.

### ۷.۲. نقش MAPIE ConformalizedQuantileRegressor و مرز آماری

طبق ADR-049، MAPIE 1.6.0 ConformalizedQuantileRegressor روی quantile estimatorهای سازگار با sklearn قرار می‌گیرد. در مسیر non-prefit، HGBR با loss="quantile" به‌عنوان estimator سازگار مستند شده است؛ در مسیر prefit، سه estimator fit‌شده به ترتیب lower، upper و median ارائه می‌شوند. MAPIE با مجموعهٔ conformalization جداگانه، بازهٔ prediction را تنظیم می‌کند. این مسیر quantileها و interval conformal می‌سازد، نه یک distribution object یا توزیع کامل NGBoost. هیچ adapter یا integration واقعی هنوز پیاده‌سازی نشده است.

لایه / مؤلفه

قرارداد / API

خروجی آماری معتبر

مرز / محدودیت

HGB quantile learner

HistGradientBoostingRegressor(loss="quantile", quantile=q)

پیش‌بینی quantile در سطح q؛ lower / median / upper

توزیع کامل، density، PDF یا CDF تولید نمی‌کند.

پارامترهای learner

scikit-learn 1.9.0؛ l2_regularization

نام پارامتر regularization در API رسمی sklearn

مقدار نهایی باید preregister/freeze شود؛ این سند مقدار عددی پیش‌فرضی تعیین نمی‌کند.

MAPIE CQR

ConformalizedQuantileRegressor 1.6.0

prediction point و بازهٔ conformal پس از conformalize

Calibration set زمانی و جدا از test؛ coverage شرطی در دادهٔ مالی تضمین نمی‌شود.

DistributionForecast

قرارداد عمومی forecast در AliX

horizon، quantile levels/values و metadata نوع خروجی

mean/dispersion فقط اگر مدل واقعاً آن‌ها را تعریف و اعتبارسنجی کند؛ از quantileها جعل نشوند.

ConformalInterval

قرارداد generic بازهٔ کالیبره‌شده

lower/upper endpoint، confidence/coverage metadata

معنای آماری خروجی مدل و generic envelope دو لایهٔ جدا هستند.

در MAPIE v1 از ConformalizedQuantileRegressor برای CQR استفاده می‌شود؛ TimeSeriesRegressor دارای methodهای مجاز مستقل خود است و method="cqr" نباید به آن نسبت داده شود.

Calibration باید زمانی و پیش از test باشد؛ split تصادفی یا calibration روی test ممنوع است. برای سری زمانی، purge/embargo و عدم هم‌پوشانی label window باید در fold plan freeze شوند.

CQR به‌خودی‌خود conditional coverage را برای بازده سری زمانی وابسته و non-stationary تضمین نمی‌کند. Temporal coverage، interval width، coverage drift، abstention و fold-level failure باید گزارش شوند؛ ادعای پوشش شرطی تضمین‌شده ممنوع است.

### ۷.۳. قاعدهٔ strict cost gate

TAKE_RESEARCH_SIGNAL(c) iff calibrated_lower_bound_gross > c / 10_000
Cost-50: calibrated_lower_bound_gross > 0.005
Cost-90: calibrated_lower_bound_gross > 0.009

مقایسه با lower endpoint پس از conformalization انجام می‌شود، نه raw HGB quantile. شرط strict است؛ برابری با هزینه pass نیست. دو سناریوی ۵۰ و ۹۰ bps باید ledger و metric مستقل داشته باشند. عبور از ۵۰ bps به معنای pass شدن ۹۰ bps نیست. قرارداد fill، fee/slippage/spread و فرض next-open/exit باید پیش از Fit تکمیل شود.

### ۷.۴. قرارداد ریاضی mock

calibrated_lower = lower_quantile - lower_correction
calibrated_upper = upper_quantile + upper_correction
require interval_ordered; 0 < coverage < 1; n_calib >= 1

### ۸. Trial Registry نسخهٔ v1.2 — ۳۰ slot تاریخی و معیارهای پذیرش

### ۸.۱. Budget و status

Family ID پژوهشی Regime R با همان N_budget=30 slot حفظ می‌شود. شناسه‌های رزروشدهٔ موجود NOBITEX-R1-NGBOOST-* به‌عنوان slot lineage تاریخی باقی می‌مانند؛ candidate جایگزین HGBQ-CQR با candidate_id مستقل، parent_candidate_id به candidate NGBoost قبلی و همان slot_id ثبت می‌شود و رابطهٔ supersedes دارد. هیچ مجموعهٔ دوم ۳۰تایی ساخته نمی‌شود.

وضعیت Trial

معنا

NOT_STARTED

شناسه رزرو شده؛ Fit یا ارزیابی رخ نداده است.

FIT_STARTED

شروع fit ثبت شده؛ لزوماً candidate اقتصادی ارزیابی نشده.

CANDIDATE_EVALUATED

خروجی candidate مشاهده/ارزیابی و در انتخاب قابل استفاده شده است.

PARTIAL / FAILED / CANCELLED

وضعیت اجرایی باید جدا از هویت candidate و attempt ثبت شود.

COMPLETED

چرخهٔ ثبت‌شده تکمیل شده؛ نتیجه و artifact hash باید موجود باشند.

### ۸.۲. حسابداری آماری

هر candidate باید candidate_id، slot_id، parent_candidate_id/supersedes در صورت جایگزینی، config hash، base-learner/model revision، quantile levels، l2_regularization، feature/target hashes، fold plan، MAPIE/CQR calibration parameters، cost/fill contract، seed، run ID و attempt ID داشته باشد.

retry، execution attempt و candidate identity جدا هستند. تغییر base learner از NGBoost به HGB candidate جدید است و باید append-only با parent link در همان ۳۰ slot supersede شود؛ رزرو slot جدیدی به بودجه اضافه نمی‌کند. مطابق ADR-045، اگر parent فقط RESERVED/NOT_STARTED بوده باشد، به‌خودی‌خود economic trial نیست؛ اگر parent به ارزیابی candidate رسیده باشد، parent و child هر دو هویت candidate مستقل و در multiplicity/DSR cohort باقی می‌مانند و حذف یا deduplicate نمی‌شوند.

GLOBAL_N=UNKNOWN و GLOBAL_DSR=NOT_COMPUTABLE باقی می‌مانند. بودجهٔ N_budget=30 فقط برای خانوادهٔ جدید sensitivity/preregistration است و جایگزین شمارش تاریخی نیست.

Family DSR فقط با تمام return seriesهای خالص، Sharpe قابل مقایسه، skewness/kurtosis و multiplicity واقعی قابل محاسبه است. اگر دادهٔ لازم ناقص باشد، عدد DSR ساخته نمی‌شود.

### ۸.۳. معیارهای پذیرش اقتصادی

معیار

Cost-50

Cost-90

سطح اعمال

Net Profit Factor

≥ 1.10

≥ 1.10

برای هر افق و هر سناریو جداگانه

Completed trades

≥ 100

≥ 100

از ledger واقعی پس از هزینه

Max Drawdown

≤ 15%

≤ 15%

از equity curve هزینه‌شده

Walk-forward

حداقل ۳ fold زمانی

حداقل ۳ fold زمانی

نتیجهٔ هر fold جداگانه حفظ شود

DSR

> 0، مشروط به feasibility

> 0، مشروط به feasibility

خانوادهٔ trial و وابستگی ثبت شوند

Forecast loss، NLL/log score، CRPS، quantile loss، interval coverage و width معیارهای تشخیصی‌اند؛ جایگزین Gate اقتصادی نیستند. metric نامعین، trade count ناکافی، fold ناقص، data leakage یا sample insufficient به معنی NO-GO/INSUFFICIENT_EVIDENCE است، نه pass.

## ۹. حاکمیت داده، Holdout و استقلال آماری

### ۹.۱. دیتاست تاریخی Nobitex pre-hack

مشخصه

مقدار مرجع

Venue / Instrument

Nobitex Public UDF BTC/USDT Spot OHLCV

Cadence

1h

Rows

31,801

Range UTC

2021-10-31T20:30:00Z to 2025-06-17T20:30:00Z

SHA-256

0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e

Integrity

Hourly grid در manifest مرجع؛ zero missing intervals گزارش شده

Holdout status

BLOCKED_CONTAMINATED برای استفاده به‌عنوان unseen confirmatory holdout

این داده در مسیرهای AI-6، DEAP و آزمایش‌های قبلی مصرف شده است. terminal slice در معرض است؛ hash کردن زیرمجموعهٔ آن یا انتخاب مرز جدید، داده را تازه و untouched نمی‌کند. استفاده از آن در R1/R2 فقط برای توسعهٔ قرارداد/تحلیل تاریخیِ آشکارا exploratory مجاز است؛ نه ادعای تأییدی OOS جدید.

### ۹.۲. Post-hack Holdout

SEALED / NO ACCESS

Post-hack Holdout همچنان sealed و خارج از مجوز این فاز است. در این نوبت هیچ دادهٔ Holdout خوانده، inspect، materialize، fetch یا استفاده نشده است. هر دسترسی آینده نیازمند دستور مستقل مالک و رعایت ADR-036 / ADR-039 / مرزهای داده است.

### ۹.۳. Venue و Source Policy

Nobitex-native data مرجع اعتبارسنجی نهایی است. Binance/Bybit فقط proxy پژوهشی‌اند و انتقال‌پذیری فرض نمی‌شود.

Manifest، SHA-256، timestamp grid، continuity، source provenance و available_at ثبت شوند. Backfill/interpolation/forward-fill برای پنهان‌کردن gap ممنوع است.

دادهٔ post-hack به‌عنوان رژیم/پنجرهٔ جداگانه ثبت می‌شود؛ نباید به‌صورت خاموش به pre-hack چسبانده شود.

هر Provider باید مجوز، شرایط استفاده، endpoint رسمی، historical coverage، rate limit، retention و دسترسی واقعی از GitHub Actions را پاس کند.

## ۱۰. Dependency، Provenance و سازگاری نسخه‌ها

### ۱۰.۱. وضعیت manifest مخزن

مورد

وضعیت مشاهده‌شده

اثر تصمیم

requirements.in

وجود ندارد.

فایل جدید نسازید؛ CI فعلی از requirements.txt استفاده می‌کند.

requirements.txt

موجود؛ dependencies مستقیم و quality tools pin شده‌اند.

Source of truth فعلی.

requirements.lock

موجود؛ pip-tools generated lock.

باید فقط از CI و ورودی فعلی بازسازی و diff شود.

CI compile command

pip-compile با input=requirements.txt و constraint= requirements.lock.

هر requirements.in جدید بدون تغییر workflow از CI جدا می‌افتد.

scikit-learn

1.9.0 در pin فعلی.

Stack هدف برای بررسی سازگاری.

NumPy

2.5.4 در requirements.lock فعلی.

Stack هدف برای بررسی resolver/API.

### ۱۰.۲. نتیجهٔ PyPI provenance و تصمیم مسیر جایگزین HGBQ-CQR

بسته / نسخه

Provenance

سازگاری اعلام‌شده / نتیجه

NGBoost 0.6.0

No provenance؛ wheel و sdist فاقد attestation.

در بررسی انجام‌شده gate زنجیرهٔ تأمین را رد می‌کند؛ برای مسیر فعال AliX انتخاب/نصب نمی‌شود.

NGBoost 0.5.11 / 0.5.10 / 0.5.9 / 0.5.8

PyPI release history برای نسخه‌های اخیر بررسی‌شده نیز No provenance.

جایگزین قابل قبول با provenance از همان بررسی پیدا نشد؛ fallback انتخاب‌شده HGBQ-CQR است.

MAPIE 1.6.0

Full provenance؛ wheel و sdist با Trusted Publishing/GitHub Actions.

شرایط اعلام‌شده Python/NumPy/scikit-learn با stack pin فعلی سازگارند؛ provenance مجوز نصب خودکار یا اثبات integration نیست.

تصمیم Dependency Gate / NGBoost 0.6.0 و نسخه‌های پایدار بررسی‌شده به‌علت فقدان PyPI provenance برای مسیر فعال رد شدند. ADR-049، HGBQ-CQR را به‌عنوان جایگزین انتخاب می‌کند: HistGradientBoostingRegressor از scikit-learn==1.9.0 موجود در lock استفاده می‌کند و MAPIE 1.6.0 provenance کامل دارد. بااین‌حال MAPIE هنوز در requirements.txt/requirements.lock نیست؛ هیچ dependency install/lock update، adapter یا integration انجام نشده و G1 تا resolution، hash/license، pip-audit و CI در GitHub Actions PASS نشود، BLOCKED باقی می‌ماند.


### ۱۰.۳. اثر بر نقشهٔ راه

گام بعدی مجاز: پس از دستور/مجوز مربوط، pin کردن MAPIE==1.6.0 در requirements.txt و بازسازی requirements.lock فقط از همان source of truth فعلی؛ NGBoost وارد dependency set نشود.

پس از انتخاب نسخه، requirements.txt و requirements.lock در یک commit محدود تغییر می‌کنند؛ requirements.in نسازید مگر workflow رسمی compile تغییر کند. scikit-learn==1.9.0 از قبل موجود است و نباید duplicate یا بی‌دلیل تغییر کند.

پیش از هر fit باید resolution، license، hash، provenance، pip-audit و compatibility در GitHub Actions بررسی شوند. MAPIE 1.6.0 provenance کامل دارد، اما این fact به‌تنهایی API compatibility، رفتار calibration در time series یا نتیجهٔ اقتصادی را اثبات نمی‌کند.

MAPIE 1.6.0 گزینهٔ قابل بررسی است؛ وجود provenance یا minimum versions به‌تنهایی adapter compatibility را اثبات نمی‌کند.

### ۱۰.۴. شناسه‌های cryptographic مشاهده‌شده

این شناسه‌ها برای بازبینی artifactهای منتشرشده ثبت شده‌اند. Checksum به‌تنهایی provenance نیست. hashهای MAPIE در بررسی قبلی به attestation متصل بوده‌اند؛ hashهای NGBoost فاقد attestation بودند و دلیل رد مسیر محسوب می‌شوند.

Artifact

SHA-256

یادداشت

MAPIE 1.6.0 wheel

6bb6b217f33f1301913d230efab68b8b6ec3c0dcdf1e7ad28ccce73435f71dfb

PyPI publish attestation؛ مشاهده‌شده در صفحهٔ release.

MAPIE 1.6.0 sdist

471631634184c4cc22641b8a72572df5b5e281e237b0c00ec6c1c668b78767c2

PyPI publish attestation؛ مشاهده‌شده در صفحهٔ release.

NGBoost 0.6.0 wheel

1fb3c5c7a2f0ebb2fe6b3c0fa1c9613b8aa33adffce7823a1493ad01b08a3c49

Checksum وجود دارد، اما provenance ندارد.

NGBoost 0.6.0 sdist

e6776b9a958e05fa150201516fe7734b2f0387799eddfdcce0b8c99cc7420c78

Checksum وجود دارد، اما provenance ندارد.

## ۱۱. CI، کیفیت مهندسی و Change Control

### ۱۱.۱. شواهد CI و Preflight تا HEAD پایهٔ تأییدشده

شاخص

مقدار

Workflow

AliX Continuous Integration

Run number / ID

#2450 / 43860221606

Commit SHA

3162f9a88efb81b7069f63afe354738df689a869

Status / conclusion

completed / success

Quality job

success

PostgreSQL integration

success

Freqtrade runtime smoke

success

Four-arm lock bare environment

success

Ensemble real-data research

skipped

Real-data diagnostic

skipped

### ۱۱.۲. Quality gates

Gate

نتیجه در Run #2450

Ruff lint

PASS

Ruff format

PASS

Mypy strict

PASS

Bandit / Security scan

PASS

Runtime/tooling dependency resolution dry-run

PASS

Transitive dependency lock verification

PASS

pip-audit dependency audit

PASS

Architecture contracts / import contracts

PASS

Xenon complexity gate

PASS

Mock-only unit tests + coverage

PASS

Standard CI #2450 روی HEAD پایه 3162f9a… موفق است. آخرین Four-Arm Quality Preflight مشاهده‌شده #46 / 43833782167 نیز SUCCESS است، اما head_sha آن e842187d1491ba50b925c4d3d300fdab6f441b7c است؛ بنابراین Preflight روی HEAD فعلی تأیید نشده. برای اجرای Preflight روی این تغییرات حاکمیتی، path filter باید docs/knowledge_base، docs/architecture/decisions و docs/governance را نیز پوشش دهد؛ patch پیشنهادی در فایل جانبی این تحویل ارائه شده است. این اصلاح فقط activation scope را توسعه می‌دهد و هیچ quality gate یا thresholdی را تغییر نمی‌دهد.

### ۱۱.۳. قواعد change control

هر تغییر حساس باید در شاخهٔ پژوهشی، با Commit محدود و پیام روشن، شواهد CI و ADR مرتبط وارد شود؛ برای تغییرات حاکمیتی، Standard CI و Quality Preflight باید روی SHA جدید بررسی شوند.

هیچ‌گاه از [skip ci] استفاده نشود؛ هیچ quality gate حذف یا relax نشود.

هر commit باید با precondition از HEAD فعلی و فایل‌های هدف تطبیق داده شود؛ stale SHA یا ref update بدون compare-and-swap ممنوع است.

همهٔ پژوهش‌ها، تست‌ها، dependency resolution و آموزش/ارزیابی مدل باید از GitHub/GitHub Actions انجام شوند. اجرای محلی پروژه یا نتیجهٔ خارج از CI، شواهد معتبر AliX محسوب نمی‌شود.

## ۱۲. ریسک، امنیت و Invariantهای غیرقابل‌مذاکره

### ۱۲.۱. ثابت‌های ایمنی

Champion = null
order_permission = false
trading_authorization = NO-GO
private_api_calls = 0
real_order_attempts = 0
orders_or_execution_api_used = false
ALIX_DRY_RUN = true
ALIX_ALLOW_LIVE_TRADING = false
authorization_status = NOT_AUTHORIZED
EXECUTION_TRIGGER = NOT VERIFIED IN LIVE TREE AUDIT (MRD v1.0 historically recorded PROBE)
ACQUISITION_TRIGGER = NOT VERIFIED IN LIVE TREE AUDIT (MRD v1.0 historically recorded PROBE)
Four-arm execution lock = NOT_AUTHORIZED
main = FROZEN @ 44bcaeae7eefd358cf063394b0e7f4f0b83af8eb
HOLDOUT_ACCESS = FORBIDDEN
GLOBAL_N = UNKNOWN
GLOBAL_DSR = NOT_COMPUTABLE

### ۱۲.۲. قواعد fail-closed

دادهٔ stale، ناقص، gapدار، schema mismatch یا با provenance نامعتبر → عدم ورود/عدم اجرا.

مدل، adapter، risk service، ledger یا permission ناموجود/ناسازگار → عدم ورود.

Unknown order status پس از timeout → ارسال مجدد یا route switch ممنوع تا reconciliation.

Absolute risk caps و Kill Switch مستقل از مدل‌اند؛ confidence=100% اجازهٔ عبور از سقف را ایجاد نمی‌کند. Reset فقط طبق مسیر manual approval.

Secretها در کد، chat، log، commit یا artifact قرار نمی‌گیرند. NOBITEX_PUBLIC_KEY فقط از GitHub Actions Secret مجاز خوانده می‌شود؛ مقدار آن نباید چاپ یا هاردکد شود.

### ۱۲.۳. Paper / Live Release

Paper Trading و Real Trading هر دو NOT AUTHORIZED هستند. سبز شدن CI، وجود adapter، اجرای workflow پژوهشی، نتیجهٔ پیش‌بینی یا پذیرش یک trial هیچ‌کدام مجوز Paper/Live ایجاد نمی‌کنند. تنها مسیر تغییر، Release Authorization جداگانهٔ مالک است که commit/config/model hashes، سقف‌های محافظتی، runbook توقف/rollback و تأیید Risk/Security را شامل شود.

#### ۱۳. نقشهٔ راه نسخهٔ v1.2 (تاریخی؛ جایگزین‌شده توسط نقشهٔ راه M8.1 در v1.3)

مسیر پیشین Regime R / HGBQ-CQR دیگر مسیر فعال مدل نیست و طبق ADR-055 بسته شده است. هیچ فاز مدل‌سازی به‌صورت خودکار فعال نمی‌شود. نسخهٔ v1.2 دو پیشنهاد جایگزین را مطرح می‌کند؛ انتخاب A یا B به مالک واگذار می‌شود و تا آن زمان وضعیت پروژه در حالت توقف پژوهش اجرایی باقی می‌ماند.

مرحله

وضعیت v1.2

دامنهٔ کار مجاز/پیشنهادی

معیار خروج

مسیرهای تاریخی HGBQ / OFI / Sentiment / Regime T

CLOSED — NO-GO

فقط نگهداری ADRها، artifactها و lineage؛ بدون احیا یا rerun.

Closure ثبت شده؛ هیچ promotion.

Option A — Target redesign

PROPOSED / NOT SELECTED

فقط طراحی target متمایز از ADR-056؛ vertical timeout صریح؛ Logistic Regression می‌تواند baseline کم‌ظرفیت باشد.

انتخاب مالک + preregistration جدید و hash‌شده؛ اجرای تجربی نیازمند مجوز مستقل.

Option B — On-chain feasibility

PROPOSED / DEFER

فقط بررسی مستند Nobitex entity coverage، PIT availability، license، API credits و کل هزینه.

انتخاب مالک + feasibility scope جدا؛ هیچ خرید/data access بدون مجوز صریح.

انتخاب فاز بعدی

PENDING OWNER DECISION

مالک فقط یکی از A یا B را انتخاب کند؛ تا آن زمان توقف.

تصمیم ثبت‌شده در ADR superseding.

Preregistration

NOT STARTED

تعریف candidate budget، target/feature, fold, purge/embargo, costs, metrics, DSR و stop rules.

Commitment hash پیش از مشاهده هر نتیجه.

Implementation / Fit / Backtest

NOT AUTHORIZED

در این MRD هیچ کد یا مدل جدید ساخته نشده و هیچ اجرا مجاز نشده است.

مجوز مستقل داده/fit/evaluation و CI روی SHA دقیق.

Paper Trading

NO-GO

ممنوع.

Release authorization مستقل و اثبات اقتصادی در آینده.

Live Trading

PROHIBITED

خارج از scope.

مجوز مستقل تازه؛ این سند هیچ اختیار نمی‌دهد.

### ۱۳.۱. گزینهٔ A — Target redesign با مدل کم‌ظرفیت

Triple-Barrier + Logistic Regression قبلاً در ADR-056 اجرا و NO-GO شده است؛ بنابراین اجرای دوبارهٔ همان تعریف هدف، همان 2×ATR14، افق 24 ساعت، برچسب دودوییِ صرفاً first-touch و حذف timeoutها مجاز یا از نظر علمی موجه نیست. هر گزینهٔ A باید یک فرضیهٔ واقعاً متفاوت و پیش‌ثبت جدید داشته باشد.

طراحی قابل‌بررسی، نه اجرای مجاز: تعریف outcome سه‌گانهٔ TP_FIRST / SL_FIRST / VERTICAL_TIMEOUT یا یک برچسب سودآوری خالص که خروج عمودی را نیز وارد کند؛ ثبت جداگانهٔ AMBIGUOUS و UNAVAILABLE؛ تعیین یک جفت barrier و یک افق ثابت پیش از هر Fit؛ ورود مرجع open(t+1) و صرفاً featureهای بسته‌شده تا close(t).

از دادهٔ OHLCV موجود فقط به‌عنوان PRE-HACK exploratory و با همان hash استفاده شود؛ ستون Volume در snapshot موجود صفر است و نباید به‌عنوان predictor فرض شود. Logistic Regression با scaler فقط fit‌شده روی train، purge کامل label windows، walk-forward زمانی و هزینه‌های Cost-50/Cost-90 حفظ شوند. معیارهای اقتصادی NetPF، trade count، MaxDD و fold stability باید از ابتدا ثبت شوند؛ accuracy به‌تنهایی کافی نیست.

تغییر هدف، horizon، barrierها، treatment از timeoutها، feature schema یا cost contract هرکدام candidate جدیدند. تعداد گزینه‌ها باید محدود و قبل از مشاهدهٔ نتایج freeze شود؛ هیچ sweep یا تغییر آستانه پس از مشاهدهٔ نتایج مجاز نیست.

هیچ Fit، inference یا backtest با این MRD مجاز نشده است. در صورت انتخاب A، گام اول فقط تدوین و hash پیش‌ثبت جدید است؛ مجوز اجرای تجربی باید جداگانه اخذ شود.

### ۱۳.۲. گزینهٔ B — Data paradigm shift به On-Chain exchange flows

ADR-061 امکان‌سنجی منبع‌ها را در ۲۷ نوامبر ۲۰۲۶ ثبت کرده است. CryptoQuant از نظر metricهای exchange-flow گزینهٔ پرداختیِ قوی‌تر است، اما Nobitex entity support در اسناد بررسی‌شده تأیید نشده و مقدارهای تاریخی بعضی از flow metrics با بازبینی wallet labels تغییر می‌کنند؛ منبع به‌عنوان Point-in-Time accurate تضمین نشده است.

Snapshot هزینهٔ ثبت‌شده در ADR-061: Basic رایگان 10,000 credits/month دارد اما on-chain ندارد؛ Advanced دادهٔ market را پوشش می‌دهد و on-chain ندارد؛ Professional اولین tier ثبت‌شده با daily on-chain access و یک سال API history است: $109 ماهانه یا $99/month با پرداخت سالانه (مجموع $1,188 سالانه). Hourly/block و full API history در آن snapshot به Premium با $799/month annual-billing نسبت داده شده است. این اعداد قیمت snapshot پژوهش‌اند، نه پیشنهاد خرید یا قیمت تضمین‌شدهٔ امروز.

ریسک بنیادی: BTC exchange inflow/outflow به‌خودی‌خود فروش BTC در بازار Nobitex Spot BTCUSDT را ثابت نمی‌کند؛ این جریان‌ها شبکه‌ای و غالباً روزانه‌اند. بدون Nobitex-labelled entity، timestamp واقعی availability و نسخه‌های point-in-time، backtest تاریخی می‌تواند leakage ناشی از اصلاح برچسب wallet داشته باشد. Glassnode Professional/API add-on قیمت عمومی ثابت نداشت؛ Coin Metrics Community غیرتجاری است و entitlement همان metric تأیید نشده است.

توصیهٔ فعلی: هیچ subscription، credit، trial یا داده‌ای خریداری یا دریافت نشود. قبل از هر انتخاب B، فقط در یک فاز امکان‌سنجی با مجوز مالک باید coverage صریح Nobitex، metric دقیق، temporal availability، retention، API credits، مجوز استفاده و هزینهٔ کل بررسی شود؛ بدون ایجاد dataset یا model.

### ۱۴.۱. Gateهای مشترک A و B

☐ انتخاب صریح مالک میان A و B؛ تا پیش از آن هیچ آزمایش تازه‌ای آغاز نشود.

☐ برای A: تعریف target و label maturity، timing، ambiguous/timeout policy، purge/embargo، feature/target hashes و هزینه‌ها باید قبل از اجرا freeze شوند.

☐ برای B: Nobitex entity coverage، point-in-time history، vendor licensing، cost/credit budget و provenance باید پیش از هر خرید یا data access اثبات شوند.

☐ بازار dependent target فقط Nobitex Spot BTCUSDT است؛ M3 OFI گذشته بر BTCIRT انجام شده و نباید به BTCUSDT تعمیم داده شود.

☐ Holdout/Post-Hack sealed؛ هر پروتکل A/B فقط روی دادهٔ مجاز exploratory و بدون ادعای confirmatory تا اخذ مجوز مستقل.

☐ Trial accounting append-only؛ تکرارهای HGBQ و candidateهای قبلی حذف یا مستقل فرض نشوند؛ GLOBAL_N همچنان UNKNOWN است.

☐ حداقل سه fold زمانی مستقل، گزارش کامل همهٔ foldها و حفظ هزینه‌های Cost-50 و Cost-90 در صورت انتخاب A.

☐ هیچ تغییر در Holdout، main، مجوزهای Paper/Live یا سازوکار risk/order به این فاز تعلق ندارد.

☐ DSR فقط در صورت تکمیل return series و multiplicity معتبر محاسبه شود؛ در غیر این صورت NOT_COMPUTABLE باقی بماند.

☐ این MRD و ADR-065 به‌تنهایی مجوز model build، fit، inference، backtest، دادهٔ جدید یا خرید API صادر نمی‌کنند.

### ۱۴.۲. معیارهای پذیرش اقتصادی

حوزه

شرط لازم

شرط رد / توقف

NetPF

≥1.10 در Cost-50 و Cost-90 برای هر افق.

کمتر از 1.10 یا metric نامعین.

Trade count

≥100 completed costed trades برای هر افق/ledger.

نمونه ناکافی؛ بدون پایین آوردن gate.

MaxDD

≤15% در هر ledger.

بیش از 15%؛ مستقل از PF.

Temporal robustness

حداقل ۳ fold؛ گزارش مستقل هر fold.

یک fold ناموفق یا unsupported پنهان نشود.

Calibration

coverage، width، drift و abstention گزارش شوند.

coverage claim غیرمجاز/عدم پایداری.

DSR / multiplicity

فقط با inputs واقعی و N قابل دفاع.

اگر infeasible، DSR ساخته نشود و evidence ناکافی اعلام شود.

قاعدهٔ No-Go

هرگونه dependency/provenance نامعتبر، leakage، calibration غیرقابل دفاع، نمونهٔ ناکافی، metric نامعین یا شکست Gate اقتصادی باید به STOP / NO-GO / INSUFFICIENT_EVIDENCE منجر شود. تغییر آستانه‌ها پس از مشاهدهٔ نتایج یا مصرف مجدد Holdout مجاز نیست.

وضعیت تصمیم: PENDING OWNER DECISION — هیچ‌یک از A یا B انتخاب نشده است. پس از اعلام انتخاب، فقط همان مسیر وارد طراحی preregistration می‌شود؛ اجرای تجربی، خرید داده یا دسترسی به Holdout نیازمند مجوز مستقل باقی می‌ماند.

## ۱۵. پروتکل ادامهٔ کار و جلوگیری از دوباره‌کاری

### ۱۵.۱. VERIFY FIRST

پیش از هر Commit یا اجرای Workflow، وضعیت زنده از GitHub بررسی شود. هیچ‌یک از مقادیر این سند به‌عنوان HEAD آینده یا مجوز اجرای آینده تلقی نشود.

HEAD پایهٔ این MRD با GitHub تطبیق داده شده است: 95133203e70265ac96f7d4dd7c14ed213fadaf38. main در این نوبت دست‌نخورده است و Tree SHA مبنای آن 5761c6c34143f8e6924f53a4e745c48932a5f791 باقی می‌ماند.

ADR-053 تا ADR-065، Trial Registry append-only و Holdout Commitment را بررسی کن. مسیرهای اقتصادی ثبت‌شده NO-GO هستند؛ هیچ وضعیت تاریخی نباید مسیر فعال تلقی شود.

requirements.txt، requirements.lock و CI فقط برای تغییر مهندسی آینده بازبینی شوند؛ در وضعیت جاری هیچ model dependency جدیدی برای HGBQ-CQR فعال نیست و مسیر آن بسته شده است.

Standard CI روی HEAD پایهٔ 95133203…، Run #46001067175، completed/success با 1,127 passed، 5 skipped و 90.36% coverage است. برای هر commit بعدی CI باید دقیقاً روی HEAD جدید بررسی شود.

هر مسیر تازه (A یا B) ابتدا به انتخاب صریح مالک و preregistration مستقل نیاز دارد؛ feasibility منبع یا سبز بودن CI مجوز خرید، استفاده یا مدل‌سازی ایجاد نمی‌کند.

هیچ training/fit/backtest/inference/data acquisition یا خرید اشتراک بدون Owner Directive و مجوز مستقل انجام نشود؛ Holdout و Post-Hack همواره sealed می‌مانند.

### ۱۵.۲. قواعد ضد دوباره‌کاری و تغییر

Commitهای قبلی و نتایج NO-GO در تاریخچه حفظ می‌شوند. HGBQ-CQR، OFI، Regime T Triple-Barrier و M5.2 Sentiment تحت ADRهای مربوطه بسته شده‌اند؛ هیچ‌کدام را با تغییر نام یا تکرار سطحی احیا نکن.

هیچ مسیر فعال HGBQ-CQR وجود ندارد؛ ADR-055 آن را بسته است. هیچ وابستگی یا adapter جدیدی برای احیای آن در این MRD مجاز نشده است.

ساخت adapter، feature، target یا مدل جدید انجام نشده است. پیشنهادهای A/B تا تصمیم مالک صرفاً گزینهٔ طراحی‌اند.

Holdout=SEALED؛ هیچ دسترسی Post-Hack، تغییر main، Paper/Live یا مجوز سفارش انجام/صادر نشده است. فقط ADR-065 و MRD v1.2 روی شاخهٔ research افزوده می‌شوند.

همهٔ ادعاهای عددی باید به Commit، Run ID، artifact یا منبع قابل بازبینی پیوند بخورند.

### ۱۵.۳. خلاصهٔ انتقال به مهندس بعدی

ACTIVE_BRANCH = research/nobitex-hf-strategy
VERIFIED_BASE_HEAD = 95133203e70265ac96f7d4dd7c14ed213fadaf38 (pre-documentation commit)
MAIN_HEAD = a1c1a658906cd79be62796885e3ed782b66a267d
MAIN_TREE = 5761c6c34143f8e6924f53a4e745c48932a5f791 (baseline tree restored; unchanged in this turn)
STANDARD_CI = #46001067175 / SUCCESS on VERIFIED_BASE_HEAD / 1,127 passed / 5 skipped / 90.36% coverage
PATHS = HGBQ-CQR CLOSED_NO_GO | M3_OFI CLOSED_NO_GO | M5.2_SENTIMENT CLOSED_NO_GO | REGIME_T_TRIPLE_BARRIER CLOSED_NO_GO
TRIALS = HGBQ 3 unique horizons (30 slots incl. 27 duplicates) | OFI 2 | Triple-Barrier 1 | M5.2 4
GLOBAL_N = UNKNOWN | GLOBAL_DSR = NOT_COMPUTABLE
NEXT_PHASE = OWNER_DECISION_PENDING: A (materially new, non-duplicative target redesign) vs B (on-chain feasibility/cost qualification)
SAFETY = Champion=null | order_permission=false | trading_authorization=NO-GO | Holdout=SEALED

## ۱۶. واژه‌نامه

اصطلاح

معنا در AliX

Regime R

جهت پژوهشی جدید؛ مسیر فعال مدل HGBQ-CQR طبق ADR-049 است؛ ADR-048 سابقهٔ انتخاب NGBoost+CQR را نگه می‌دارد.

NGBoost

مدل gradient boosting توزیعی که در ADR-048 انتخاب تاریخی شد؛ به‌علت فقدان PyPI provenance برای مسیر فعال رد شده است.

CQR

Conformalized Quantile Regression؛ اصلاح quantileها با دادهٔ کالیبراسیون مستقل زمانی.

MAPIE

کتابخانهٔ prediction intervals؛ API موردنظر CQR، ConformalizedQuantileRegressor است.

DistributionForecast

قرارداد عمومی forecast برای horizon، quantile levels/values و metadata نوع خروجی؛ mean/dispersion فقط در صورت تعریف آماری معتبر توسط مدل.

ConformalInterval

قرارداد mock برای بازهٔ تنظیم‌شده و مشخصات calibration.

Lower-bound gate

پذیرش سیگنال پژوهشی فقط در صورت عبور strict lower endpoint از cost.

Trial ID

شناسهٔ رزروشدهٔ candidate؛ به‌تنهایی نشانهٔ اجرا یا ارزیابی نیست.

Provenance

Attestation قابل‌راستی‌آزمایی از منشأ/نحوهٔ انتشار artifact بسته.

Holdout sealed

دادهٔ نهایی خارج از دسترس پژوهش جاری؛ باز شدن آن نیازمند مجوز صریح است.

Fail-closed

در نبود داده/مدل/مجوز معتبر، رفتار پیش‌فرض عدم ورود یا توقف است.

HGBQ-CQR

سه quantile با HistGradientBoostingRegressor(loss="quantile") و بازهٔ conformal با MAPIE؛ quantile + interval، نه توزیع کامل.

l2_regularization

نام پارامتر regularization در HistGradientBoostingRegressor؛ مقدار باید در preregistration تعیین و freeze شود و در این سند مقدار پیش‌فرضی تثبیت نشده است.

### ۱۶.۱. منابع داخلی مخزن و شواهد اجرایی

[S1] ADR-047 — Regime R revival strategic pivot https://github.com/angry0tiger-sudo/AliX/blob/research/nobitex-hf-strategy/docs/architecture/decisions/ADR-047-regime-r-revival-strategic-pivot.md

[S2] ADR-048 — Regime R NGBoost + CQR (historical decision, superseded for active dependency path) https://github.com/angry0tiger-sudo/AliX/blob/research/nobitex-hf-strategy/docs/architecture/decisions/ADR-048-regime-r-ngboost-cqr.md

[S7] HEAD مرجع v1.2 قبل از این commit: https://github.com/angry0tiger-sudo/AliX/commit/95133203e70265ac96f7d4dd7c14ed213fadaf38

[S4] Historical Trial Registry https://github.com/angry0tiger-sudo/AliX/blob/research/nobitex-hf-strategy/docs/research/TRIAL_REGISTRY.md

[S5] Holdout Commitment https://github.com/angry0tiger-sudo/AliX/blob/research/nobitex-hf-strategy/docs/research/HOLDOUT_COMMITMENT.md

[S6] Standard CI #2450 / Run 43860221606 https://github.com/angry0tiger-sudo/AliX/actions/runs/43860221606

[S7] HEAD مرجع v1.2 قبل از این commit: https://github.com/angry0tiger-sudo/AliX/commit/95133203e70265ac96f7d4dd7c14ed213fadaf38

[S8] ADR-049 — HGBQ-CQR dependency resolution and fallback governance https://github.com/angry0tiger-sudo/AliX/blob/research/nobitex-hf-strategy/docs/architecture/decisions/ADR-049-hgbq-cqr-dependency-resolution-and-fallback-governance.md

[S9] ADR-069 — R&D Strategy Pivot https://github.com/angry0tiger-sudo/AliX/blob/research/nobitex-hf-strategy/docs/research/ADR-069-rd-strategy-pivot.md
[S10] ADR-070 — M7 Cross-Sectional Owner Selection https://github.com/angry0tiger-sudo/AliX/blob/research/nobitex-hf-strategy/docs/adr/ADR-070-m7-cross-sectional-owner-selection.md
[S11] ADR-072 — M7 G2.0 Feature-Quality NO-GO https://github.com/angry0tiger-sudo/AliX/blob/research/nobitex-hf-strategy/docs/adr/ADR-072-m7-cross-sectional-feature-quality-no-go.md
[S12] ADR-073 — M7 G3.0 Feature-Only Supersession https://github.com/angry0tiger-sudo/AliX/blob/research/nobitex-hf-strategy/docs/adr/ADR-073-m7-feature-availability-diagnostic.md
[S13] ADR-074 — Formal M7 Lifecycle Closure https://github.com/angry0tiger-sudo/AliX/blob/research/nobitex-hf-strategy/docs/adr/ADR-074-m7-cross-sectional-lifecycle-closure.md
[S14] ADR-075 — M8 Data Market Eligibility / BTCIRT mismatch https://github.com/angry0tiger-sudo/AliX/blob/research/nobitex-hf-strategy/docs/adr/ADR-075-m8-microstructure-data-market-eligibility.md
[S15] Trial Registry (M6/M7 append-only entries) https://github.com/angry0tiger-sudo/AliX/blob/research/nobitex-hf-strategy/docs/research/TRIAL_REGISTRY.md
[S16] M8.1 Capture Run 46106438168 (capture job success; raw Artifact 15249319923) https://github.com/angry0tiger-sudo/AliX/actions/runs/46106438168
[S17] M8.1 reanalysis Run 46113673381 (successful SHA audit, OBI analysis, ADR-076 commit and analysis Artifact 15250428572) https://github.com/angry0tiger-sudo/AliX/actions/runs/46113673381
[S18] ADR-076 — M8.1 BTCUSDT One-Hour Public L2 Capture and Correlation https://github.com/angry0tiger-sudo/AliX/blob/research/nobitex-hf-strategy/docs/adr/ADR-076-m8-1-btcusdt-capture-and-correlation.md


[W5] MAPIE 1.6.0 API — ConformalizedQuantileRegressor https://mapie.readthedocs.io/en/stable/api/regression/

[W6] scikit-learn — HistGradientBoostingRegressor (loss="quantile", quantile, l2_regularization) https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html

### ۱۶.۲. منابع بیرونی Dependency و provenance

[W1] PyPI — NGBoost project / release 0.6.0; latest stable as checked 2026-11-16; no provenance https://pypi.org/project/ngboost/

[W2] PyPI — MAPIE 1.6.0; full provenance and dependency minimums https://pypi.org/project/MAPIE/1.6.0/

[W3] PyPI documentation — provenance/attestation semantics https://docs.pypi.org/using-pypi/package-security/provenance/

[W4] NGBoost GitHub release notes https://github.com/stanfordmlgroup/ngboost/releases

منابع PyPI و مستندات رسمی در ۱۶ نوامبر ۲۰۲۶ برای این نسخه بازبینی شده‌اند. Provenance منشأ artifact را قابل‌راستی‌آزمایی‌تر می‌کند، اما به‌تنهایی ایمنی کد، API compatibility، کیفیت علمی، اعتبار calibration یا مناسب‌بودن اقتصادی strategy را تضمین نمی‌کند.

## پیوست A. فهرست کامل ۳۰ Trial ID

این ۳۰ شناسهٔ تاریخی در قالب پیش‌ثبت ثبت شده‌اند. status فعلی آن‌ها باید از Trial Registry زنده خوانده شود؛ این فهرست به معنی ۳۰ Fit یا ۳۰ ارزیابی انجام‌شده نیست. برای HGBQ-CQR همان ۳۰ slot استفاده می‌شوند؛ candidate child باید با parent_candidate_id/supersedes در همان slot append شود و شناسهٔ قبلی پاک یا بازنویسی نشود.

شناسهٔ اول

شناسهٔ دوم

شناسهٔ سوم

NOBITEX-R1-NGBOOST-1h-01

NOBITEX-R1-NGBOOST-1h-02

NOBITEX-R1-NGBOOST-1h-03

NOBITEX-R1-NGBOOST-1h-04

NOBITEX-R1-NGBOOST-1h-05

NOBITEX-R1-NGBOOST-1h-06

NOBITEX-R1-NGBOOST-1h-07

NOBITEX-R1-NGBOOST-1h-08

NOBITEX-R1-NGBOOST-1h-09

NOBITEX-R1-NGBOOST-1h-10

NOBITEX-R1-NGBOOST-4h-01

NOBITEX-R1-NGBOOST-4h-02

NOBITEX-R1-NGBOOST-4h-03

NOBITEX-R1-NGBOOST-4h-04

NOBITEX-R1-NGBOOST-4h-05

NOBITEX-R1-NGBOOST-4h-06

NOBITEX-R1-NGBOOST-4h-07

NOBITEX-R1-NGBOOST-4h-08

NOBITEX-R1-NGBOOST-4h-09

NOBITEX-R1-NGBOOST-4h-10

NOBITEX-R1-NGBOOST-24h-01

NOBITEX-R1-NGBOOST-24h-02

NOBITEX-R1-NGBOOST-24h-03

NOBITEX-R1-NGBOOST-24h-04

NOBITEX-R1-NGBOOST-24h-05

NOBITEX-R1-NGBOOST-24h-06

NOBITEX-R1-NGBOOST-24h-07

NOBITEX-R1-NGBOOST-24h-08

NOBITEX-R1-NGBOOST-24h-09

NOBITEX-R1-NGBOOST-24h-10

هر slot باید candidate_id، slot_id، parent_candidate_id/supersedes (در صورت جایگزینی)، config hash، base learner/model revision، feature/target hash، fold plan hash، quantile levels، l2_regularization، MAPIE/CQR calibration parameters، cost/fill contract، seed، run ID، attempt ID و status را append-only ثبت کند. رزرو slot به‌تنهایی N مؤثر نیست؛ هر parent یا child که به ارزیابی candidate برسد، طبق ADR-045 در multiplicity باقی می‌ماند.


## پیوست B. تاریخچهٔ CI و معیارهای عبورشده

Run

SHA

علت / نتیجه

#2448 / 43833782132

e842187d1491ba50b925c4d3d300fdab6f441b7c

FAIL؛ Xenon در validate_distribution_forecast روی contracts.py با Rank C.

#2449 / 43836687806

d98df5ea57e28e1420668983afba586874ed5d6f

FAIL؛ Ruff format یک blank line اضافه می‌خواست.

#2450 / 43860221606

3162f9a88efb81b7069f63afe354738df689a869

SUCCESS؛ تمام چهار job اصلی موفق؛ دو job پژوهش real-data skipped.

Preflight #46 / 43833782167

e842187d1491ba50b925c4d3d300fdab6f441b7c

SUCCESS؛ روی SHA قدیمی‌تر اجرا شد، نه روی HEAD پایه 3162f9a…؛ شواهد مستقیم preflight روی HEAD پایه موجود نیست.

B.1. توالی اصلاح

e842187: قالب‌بندی فایل تست و اصلاح دادهٔ تست CQR برای ایجاد واقعاً interval وارونه؛ contracts.py در این commit دست‌نخورده بود.

d98df5e: تقسیم validate_distribution_forecast به helperهای خصوصی برای کاهش complexity؛ منطق، ترتیب خطاها و امضای public حفظ شد.

3162f9a: یک blank line اضافه شد؛ diff دقیقاً یک addition و صفر deletion در contracts.py بود؛ Ruff format و Xenon اکنون PASS هستند.

B.2. محدودهٔ PASS و NOT PROVEN

مورد

وضعیت

Ruff / Mypy / Bandit / Xenon / Architecture / Unit tests

PASS در Standard CI #46001067175 روی SHA 95133203…؛ 1,127 passed، 5 skipped، coverage=90.36%.

Dependency resolution / lock / pip-audit

PASS در Standard CI #46001067175؛ فقط وضعیت سلامت مهندسی dependency set فعلی را نشان می‌دهد.

PostgreSQL / Freqtrade smoke / four-arm lock

PASS در Standard CI #46001067175.

HGBQ-CQR / G4.6

CLOSED NO-GO؛ هیچ lower bound از هزینه عبور نکرد؛ volume ablation بهبود نداد.

M3 OFI

CLOSED_NO_GO؛ 2/2 Trial رد؛ دادهٔ تاریخی BTCIRT، نه BTCUSDT.

Regime T Triple-Barrier

CLOSED NO-GO؛ accuracy=51.09% < 55% preregistered screen.

M5.2 Sentiment

CLOSED NO-GO؛ 4/4 Trial رد؛ Sentiment در 4/4 comparison بدتر از Control.

Paper / Live / Holdout

NO-GO / NO ACCESS؛ Holdout=SEALED؛ مجوز سفارش صادر نشده است.
Next step

A/B پیشنهاد شده؛ هیچ‌کدام انتخاب یا مجاز نشده؛ انتظار تصمیم مالک.

AliX وضعیت v1.2 را ثبت می‌کند: HGBQ-CQR، M3 OFI و M5.2 Sentiment با حکم قطعی NO-GO بسته شده‌اند؛ Regime T Triple-Barrier قبلی نیز به‌علت عدم عبور از آستانهٔ accuracy بسته است. Trial inventory و تفاوت BTCIRT/BTCUSDT حفظ شده‌اند. Standard CI روی SHA پایهٔ مستندات موفق است. گزینه‌های A و B صرفاً پیشنهاد هستند؛ انتخاب به مالک واگذار می‌شود. هیچ مدل جدید، دادهٔ جدید یا دسترسی Holdout مجاز نشده است؛ Champion=null، order_permission=false، trading_authorization=NO-GO و Holdout=SEALED باقی می‌مانند.

AliX | Architecture First, Evidence Always

### پیوست C. افزودهٔ Trial Registry و وضعیت M8.1 در نسخهٔ ۱٫۳

C.1. Regime M6 — Three-Class Triple-Barrier

- دو identity ثبت‌شده: `NOBITEX-M6-1h-RF-3CLASS-01` و `NOBITEX-M6-4h-RF-3CLASS-01`.
- 1h: 1,177 معامله؛ NetPF=0.160856 در Cost-50 و 0.013034 در Cost-90؛ MaxDD=22.58% و 41.29%؛ 2/4 fold معتبر و 0/4 پایدار.
- 4h: 501 معامله؛ NetPF=0.534557 در Cost-50 و 0.278479 در Cost-90؛ MaxDD=28.18% و 56.06%؛ 2/4 fold معتبر و 0/4 پایدار.
- نتیجهٔ هر دو: `INSUFFICIENT_SAMPLE / NO-GO`. اجرای ناموفق اولیه که پیش از خواندن داده/Fit متوقف شد، Trial اقتصادی مستقل محسوب نمی‌شود.

C.2. M7 Cross-Sectional Ridge

- G1.2: `NOBITEX-M7-5PAIR-RIDGE-TOP2-01` و `NOBITEX-M7-10PAIR-RIDGE-TOP2-01`؛ هر دو `BLOCKED_DATA` و بدون Fit.
- G2.0: `NOBITEX-M7-5PAIR-RIDGE-TOP2-02` و `NOBITEX-M7-9PAIR-RIDGE-TOP2-01`؛ هر دو به دلیل `m7_panel_empty_after_feature_qualification` مسدود شدند؛ Fit انجام نشد.
- G3.0: همان دو identity G2.0 با revision ویژگی `log_return_24h` ارزیابی شدند. هر چهار ترکیب candidate/cost، NetPF<1 و MaxDD بسیار بالاتر از حد پذیرش داشتند؛ ۰/۴ fold پایدار برای هر candidate. M7 رسماً `CLOSED_NO_GO` است.
- تعداد identityهای متمایز M7 در G1.2/G2.0 برابر چهار است؛ دو identity G2.0 در G3.0 با config revision جداگانه دوباره ارزیابی شده‌اند و این revision در multiple-testing lineage باقی می‌ماند.

C.3. M8 source gate و M8.1 run snapshot

- M1.6 قدیمی: پنج Artifact عمومی، تماماً `BTCIRT`؛ 171,604 order-book rows، 25,623 trade rows و 8 gap record. برای فرضیه BTCUSDT واجد صلاحیت نیست.
- M8: `BLOCKED`؛ همبستگی‌های OBI برای BTCUSDT از داده‌های M1.6 محاسبه نشده‌اند و باید `NOT_COMPUTED` باقی بمانند.
- M8.1 Capture: Run `46106438168` روی SHA `a0b550f23071f8bf920b866771eb75cba30d8bf6`؛ Job Capture موفق؛ مدت واقعی ۶۰٫۰۸۸ دقیقه؛ ۲٬۵۷۹ book / ۹۳ trades / صفر gap.
- Raw Artifact: ID `15249319923`؛ digest `sha256:bc16983be51490cafebff11c5de06360d6668633bd3a75ea1afbe0defd3837c0`؛ Manifest SHA `29fe9d3a556d810fc441da9401f529d8efae0c8c507dce88b4a861c69e2a8107`.
- Analyzer Run `46113673381` با نتیجهٔ `SUCCESS`؛ Artifact تحلیل ID `15250428572`؛ ADR-076 ثبت شد.
- OBI top-5: ۵m N=8، Pearson r=0.47108390، p دوطرفه=0.23869631، Bonferroni p=1.0؛ Spearman rho=0.26190476، Bonferroni p=1.0. افق ۱۵m: N=2 و correlation estimable نیست.
- تصمیم تشخیصی `NO_SIGNIFICANT_OBI_CORRELATION` است؛ نه صفر همبستگی اثبات شده و نه حکم اقتصادی NO-GO. یک ساعت داده به‌دلیل N بسیار کوچک فقط اکتشافی است.
- `TRIGGER_M8`، `TRIGGER_M8_ANALYSIS` و Push triggerهای موقت حذف شده‌اند؛ Workflowهای باقی‌مانده فقط `workflow_dispatch` را حفظ می‌کنند.

C.4. Invariantهای غیرقابل‌مذاکره

`Champion=null` | `order_permission=false` | `trading_authorization=NO-GO`

`Holdout=SEALED` | `HOLDOUT_ACCESS=FORBIDDEN` | `posthack_data_read=false`

`private_api_calls=0` | `real_order_attempts=0` | `paper_orders=0`

`GLOBAL_N=UNKNOWN` | `GLOBAL_DSR=NOT_COMPUTABLE`

این پیوست هیچ candidate اقتصادی جدید، مجوز Fit، مجوز Paper/Live یا دسترسی Holdout ایجاد نمی‌کند. Capture دادهٔ عمومی M8.1 به خودی خود Trial اقتصادی محسوب نمی‌شود.