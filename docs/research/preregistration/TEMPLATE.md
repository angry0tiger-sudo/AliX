# AliX — Research Preregistration Template

**قانون:** این فرم باید پیش از اولین fit، tuning، انتخاب variant یا ارزیابی اقتصادی تکمیل و امضای مالک شود. پس از شروع trial، این سند immutable است؛ تغییر hypothesis یا معیارها نیازمند شناسه و ثبت‌نام جدید است.

## A. شناسه و ثبت
- شناسه فرضیه (Hypothesis ID):
- شناسه Trial / Trial Family:
- نسخه ثبت‌نام:
- تاریخ و ساعت ثبت (UTC):
- شاخه و SHA کد:
- مسیر Workflow و SHA فایل Workflow:
- دستور مالک / ADR:
- مسئول اجرا:
- امضای مالک (نام، شناسه/تاریخ امضا، UTC): **اجباری**

## B. فرضیه و هدف
- فرضیه قابل ابطال (یک جمله):
- فرض صفر:
- فرض جایگزین:
- سازوکار اقتصادی مورد انتظار:
- Target:
- Target semantics (دقیقاً مقدار/رویدادی که برچسب می‌زند، زمان قابل‌دسترس‌شدن، جهت/class، نحوه برخورد با tie/expiry):
- افق پیش‌بینی (horizon، به تعداد کندل و مدت زمانی):
- زمان تصمیم:
- زمان/قیمت فرضی اجرا:
- تعریف خروج و حداکثر مدت نگهداری:
- شرط ابطال فرضیه و تغییرات پس از مشاهده نتایج که ممنوع‌اند:

## C. بازار و داده
- Venue / Market / Symbol:
- جهت مجاز: Spot long-only:
- تایم‌فریم: **فقط یکی از 15m / 30m / 1h / 4h**:
- Provider و endpoint:
- بازه UTC شروع (inclusive):
- بازه UTC پایان (inclusive یا exclusive؛ صریح مشخص شود):
- نام فایل و مسیر artifact:
- تعداد ردیف مورد انتظار / واقعی:
- CSV SHA-256:
- Manifest SHA-256:
- شبکه timestamp و timezone:
- تعداد gap و duplicate:
- سیاست gap: FAIL-CLOSED؛ هیچ backfill، interpolation، forward-fill، synthetic bar، masking یا splice مجاز نیست:
- منبع داده و محدودیت authority (public market data، نه account/order/fill truth):
- Run ID تولید داده:

## D. تقسیم زمانی و ضد نشت
- نوع split:
- مرز دقیق train (UTC و row):
- مرز دقیق validation (UTC و row):
- مرز final holdout (UTC و row):
- تعداد foldهای زمانی (حداقل ۳):
- طول purge:
- طول embargo:
- افق label مورد استفاده برای purge/embargo:
- scaler فقط روی train: YES/NO:
- ویژگی‌ها strictly past-only: YES/NO:
- as-of alignment rule:
- final holdout CSV/slice SHA-256:
- تاریخ commitment (UTC):
- سابقه exposure به holdout (YES/NO/UNKNOWN + شواهد):
- هر نوع exposure قبلی یا UNKNOWN => holdout برای ادعای untouched OOS نامعتبر است.

## E. مدل و وزن‌ها
برای هر بازو:
- مدل و variant دقیق:
- revision / immutable model revision:
- مسیر weight و SHA-256 وزن‌ها:
- نسخه runtime/library:
- context length:
- تعداد warm-up کندل حذف‌شده:
- earliest feature timestamp:
- as-of availability timestamp:
- corpus پیش‌آموزش و منبع provenance:
- همپوشانی corpus پیش‌آموزش با پنجره بک‌تست (مقدار/درصد/بازه):
- شاهد و عدم‌قطعیت overlap:
- نبود/خرابی artifact => FAIL-CLOSED.

## F. طراحی آزمایش و variantها
- Baseline:
- بازوهای درمان:
- تنها تفاوت مجاز بین بازوها:
- row set مشترک:
- fold boundaries مشترک:
- seed یا seedهای دقیق:
- نسخه و SHA شبیه‌ساز:
- سیاست fill/stop/collision مشترک:
- سیاست risk/position مشترک:
- تعداد variantهای برنامه‌ریزی‌شده (عدد صحیح):
- فهرست شناسه تمام variantها:
- inner-CV / random search / hyperparameter variants و شمارش آن‌ها:
- Trial Registry IDs مربوط به multiple testing:

## G. قرارداد هزینه (هر دو اجباری)
| سناریو | هزینه round-trip | decomposition maker/taker و slippage به ازای هر سمت | الزام |
|---|---:|---|---|
| Cost-50 | 50 bps | دقیقاً ثبت شود | اجباری |
| Cost-90 | 90 bps | دقیقاً ثبت شود | اجباری |

نتایج دو سناریو جداگانه گزارش شوند؛ هیچ میانگین‌گیری یا جایگزینی خاموش مجاز نیست. فرض post-only guarantee برای Nobitex ممنوع است.

## H. دروازه‌های پذیرش مطلق
- Net Profit Factor پس از هزینه >= 1.10:
- completed trades >= 100:
- Max Drawdown <= 15%:
- DSR Z > 0 با N کامل همه trialهای مرتبط:
- پایداری در حداقل ۳ fold زمانی:
- هزینه‌های 50 و 90 bps هر دو گزارش شده باشند:
- بهبود نسبت به baseline زیان‌ده، به‌تنهایی pass نیست:
- fold ناقص، sample ناکافی، metric تعریف‌نشده یا provenance نامعتبر => NO-GO / INSUFFICIENT_EVIDENCE.

## I. طرح آماری و توقف
- Primary metric:
- Secondary metrics:
- روش DSR و assumptions:
- روش PBO و assumptions:
- N دقیق و فهرست source trialهای شمارش‌شده:
- وابستگی/همبستگی trialها:
- confidence sequence / e-value در ارزیابی sequential:
- stopping rule:
- multiple-comparison correction:
- شیوه شمارش failed/cancelled/partial/skipped trials:

## J. بازتولید و ایمنی
- Lockfile SHA:
- Freqtrade: 2026.7 pinned:
- Python: 3.12:
- GitHub Actions run ID:
- artifact IDs و SHA-256 خروجی‌ها:
- Champion=null:
- order_permission=false:
- trading_authorization=NO-GO:
- ALIX_DRY_RUN=true:
- ALIX_ALLOW_LIVE_TRADING=false:
- private_api_calls=0:
- real_order_attempts=0:
- بدون secret در source/log/artifact: CONFIRMED

## K. تأیید و قفل
- مالک/بازبین:
- تاریخ بازبینی UTC:
- وضعیت approval/rejection:
- SHA-256 فایل preregistration:
- اولین Run ID:
- deviations (باید NONE باشد؛ هر deviation نیازمند preregistration revision جدید است):
- امضای نهایی مالک:
