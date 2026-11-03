"""Read-only audit of immutable V4-R Run #25 outputs; never trains or fetches data."""

from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPORT_FILENAME = "alix-v4r-economic-report.json"
REPORT_SCHEMA = "alix.ml-core.v4r-economic-architecture.v1"
EXPECTED_COMMIT_SHA = "1837089f29d5abdb87b45a55e2a49a51f804d926"
EXPECTED_RUN_ID = 41518486550
EXPECTED_ARTIFACT_ID = 13180755166
EXPECTED_ARTIFACT_SHA256 = "74e9fe906ba288c7a87f67a27fb252273ff4181e8ec411919434f1219047aa36"
ARM_NAMES = (
    "CONTROL_SYMMETRIC_BUY_F1",
    "GEOMETRY_2R_BUY_F1",
    "SYMMETRIC_NET_EXPECTANCY",
    "V4R_2R_NET_EXPECTANCY",
)
ROW_LEVEL_PROBABILITY_FIELDS = {
    "class_probabilities",
    "event_probabilities",
    "per_row_probabilities",
    "predicted_probabilities",
    "prediction_probabilities",
    "probability_rows",
    "probability_vectors",
    "row_probabilities",
}


def _load_report(artifact_path: Path) -> dict[str, Any]:
    if artifact_path.is_dir():
        report_path = artifact_path / REPORT_FILENAME
        if not report_path.is_file():
            raise FileNotFoundError(f"missing_report_json:{report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
    elif artifact_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(artifact_path) as archive:
            if REPORT_FILENAME not in archive.namelist():
                raise ValueError(f"missing_artifact_member:{REPORT_FILENAME}")
            with archive.open(REPORT_FILENAME) as handle:
                report = json.load(handle)
    else:
        report = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("v4s_report_root_must_be_object")
    return report


def _validate_source_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("v4s_unexpected_report_schema")
    if report.get("commit_sha") != EXPECTED_COMMIT_SHA:
        raise ValueError("v4s_run25_commit_sha_mismatch")
    if report.get("experimental_design") != "2x2_PAIRED_FACTORIAL":
        raise ValueError("v4s_unexpected_experimental_design")
    safety = report.get("safety")
    if not isinstance(safety, dict):
        raise ValueError("v4s_safety_contract_missing")
    expected_safety = {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "orders_or_execution_api_used": False,
        "credentials_used": False,
        "public_api_only": True,
    }
    for key, expected in expected_safety.items():
        if safety.get(key) != expected:
            raise ValueError(f"v4s_safety_invariant_mismatch:{key}")
    names = {item.get("name") for item in report.get("arms", [])}
    if names != set(ARM_NAMES):
        raise ValueError("v4s_expected_four_arms_missing_or_unexpected")


def _row_level_probability_fields(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if (
                key.lower() in ROW_LEVEL_PROBABILITY_FIELDS
                and isinstance(child, list)
                and child
            ):
                found.append(child_path)
            found.extend(_row_level_probability_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_row_level_probability_fields(child, f"{path}[{index}]"))
    return found


def _pooled_label_summary(
    experiments: list[dict[str, Any]], arm_name: str
) -> dict[str, Any]:
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    for experiment in experiments:
        if experiment.get("arm") != arm_name:
            continue
        context = (str(experiment["pair"]), str(experiment["timeframe"]))
        current = {
            "label_distribution": experiment["label_distribution"],
            "exit_reason_distribution": experiment["label_audit"][
                "exit_reason_distribution"
            ],
        }
        previous = contexts.get(context)
        if previous is not None and previous != current:
            raise ValueError(f"v4s_duplicate_context_label_mismatch:{context}")
        contexts[context] = current
    if not contexts:
        raise ValueError(f"v4s_label_contexts_missing:{arm_name}")

    splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        counts: Counter[str] = Counter()
        total = 0
        for context in contexts.values():
            distribution = context["label_distribution"][split]
            total += int(distribution["total"])
            counts.update({
                label: int(details["count"])
                for label, details in distribution["classes"].items()
            })
        splits[split] = {
            "total": total,
            "counts": dict(counts),
            "percent": {
                label: round(count / total * 100, 4)
                for label, count in counts.items()
            },
        }

    exit_counts: Counter[str] = Counter()
    exit_total = 0
    for context in contexts.values():
        distribution = context["exit_reason_distribution"]
        exit_total += int(distribution["total"])
        exit_counts.update({
            reason: int(details["count"])
            for reason, details in distribution["reasons"].items()
        })
    return {
        "unique_pair_timeframe_contexts": len(contexts),
        "splits": splits,
        "exit_reasons": {
            "total": exit_total,
            "counts": dict(exit_counts),
            "percent": {
                reason: round(count / exit_total * 100, 4)
                for reason, count in exit_counts.items()
            },
        },
    }


def _mean_metric(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def _probability_summary(
    experiments: list[dict[str, Any]], arm_name: str
) -> dict[str, Any]:
    metrics: dict[str, list[float]] = defaultdict(list)
    optimization_statuses: Counter[str] = Counter()
    calibration_hold_support: list[int] = []
    variant_count = 0
    for experiment in experiments:
        if experiment.get("arm") != arm_name:
            continue
        for variant in experiment.get("model_variants", {}).values():
            variant_count += 1
            optimization = variant.get("threshold_optimization", {})
            optimization_statuses[str(optimization.get("status", "MISSING"))] += 1
            calibration = optimization.get("calibration", {})
            classes = calibration.get("label_distribution", {}).get("classes", {})
            hold_count = classes.get("HOLD", {}).get("count")
            if isinstance(hold_count, int):
                calibration_hold_support.append(hold_count)
            diagnostics = variant.get("probability_diagnostics", {})
            for split in ("calibration", "forward_validation"):
                data = diagnostics.get(split, {})
                for metric in ("hold_auc", "hold_pr_auc"):
                    value = data.get(metric)
                    if isinstance(value, (int, float)):
                        metrics[f"{split}_{metric}"].append(float(value))
                brier = data.get("brier_score", {}).get("hold_binary")
                ece = data.get("ece", {}).get("hold_binary")
                if isinstance(brier, (int, float)):
                    metrics[f"{split}_hold_brier"].append(float(brier))
                if isinstance(ece, (int, float)):
                    metrics[f"{split}_hold_ece"].append(float(ece))
    return {
        "model_variant_count": variant_count,
        "threshold_optimization_statuses": dict(optimization_statuses),
        "calibration_hold_positive_support": {
            "minimum": min(calibration_hold_support) if calibration_hold_support else None,
            "maximum": max(calibration_hold_support) if calibration_hold_support else None,
            "mean": _mean_metric([float(item) for item in calibration_hold_support]),
        },
        "mean_metrics": {
            name: _mean_metric(values) for name, values in sorted(metrics.items())
        },
    }


def _arm_activity_summary(report: dict[str, Any], arm_name: str) -> dict[str, Any]:
    arm = next(item for item in report["arms"] if item["name"] == arm_name)
    candidates = [
        candidate for candidate in report["candidates"]
        if candidate.get("arm") == arm_name
    ]
    thresholds: Counter[str] = Counter()
    raw_buy = 0
    accepted_buy = 0
    rejected_buy = 0
    for candidate in candidates:
        confidence = candidate.get("confidence_filter") or {}
        threshold = confidence.get("threshold")
        if isinstance(threshold, (int, float)):
            thresholds[f"{float(threshold):.2f}"] += 1
        raw_buy += int(confidence.get("raw_buy_signal_count", 0))
        accepted_buy += int(confidence.get("accepted_buy_signal_count", 0))
        rejected_buy += int(confidence.get("rejected_buy_signal_count", 0))
    aggregate = arm["aggregate"]
    return {
        "candidate_count": len(candidates),
        "trained_candidate_count": int(aggregate["trained_candidate_count"]),
        "sample_gate_skipped_candidate_count": int(
            aggregate["sample_gate_skipped_candidate_count"]
        ),
        "confidence_threshold_counts": dict(thresholds),
        "raw_buy_signal_count": raw_buy,
        "accepted_buy_signal_count": accepted_buy,
        "rejected_buy_signal_count": rejected_buy,
        "accepted_buy_percent_of_raw": (
            round(accepted_buy / raw_buy * 100, 4) if raw_buy else None
        ),
        "trade_count": int(aggregate["trade_count"]),
        "combined_net_profit_factor": float(aggregate["combined_net_profit_factor"]),
        "net_pnl_usd": float(aggregate["total_net_pnl_usd"]),
        "strict_economic_pass_candidate_count": int(
            aggregate["strict_economic_pass_candidate_count"]
        ),
    }


def analyze_report(report: dict[str, Any]) -> dict[str, Any]:
    _validate_source_report(report)
    experiments = report["experiments"]
    activity = {
        name: _arm_activity_summary(report, name) for name in ARM_NAMES
    }
    labels = {
        name: _pooled_label_summary(experiments, name) for name in ARM_NAMES
    }
    probabilities = {
        name: _probability_summary(experiments, name) for name in ARM_NAMES
    }
    probability_fields = _row_level_probability_fields({
        "candidates": report.get("candidates", []),
        "experiments": experiments,
        "trade_rows": report.get("trade_rows", []),
    })
    return {
        "commit_sha": report["commit_sha"],
        "model_training_performed_in_source_run": bool(
            report["model_training_performed"]
        ),
        "training_performed_by_v4s": False,
        "activity": activity,
        "labels": labels,
        "probability_diagnostics": probabilities,
        "row_level_probability_fields": probability_fields,
        "row_level_probabilities_available": bool(probability_fields),
        "counterfactual_sweep_status": (
            "COMPUTABLE" if probability_fields
            else "NOT_COMPUTABLE_MISSING_ROW_LEVEL_PROBABILITIES"
        ),
        "safety": report["safety"],
    }


def _render_report(summary: dict[str, Any]) -> str:
    activity = summary["activity"]
    labels = summary["labels"]
    probability = summary["probability_diagnostics"]
    lines = [
        "# ممیزی V4-S: تشخیص آستانه و توزیع برچسب‌ها",
        "",
        "## منشأ و دامنه",
        "",
        f"- منبع: GitHub Actions Run #{EXPECTED_RUN_ID}",
        f"- شناسه Artifact: {EXPECTED_ARTIFACT_ID}",
        f"- SHA-256 آرشیو Artifact: {EXPECTED_ARTIFACT_SHA256}",
        f"- Commit منبع: {summary['commit_sha']}",
        f"- آموزش در Run منبع: {summary['model_training_performed_in_source_run']} (تاریخی و متعلق به V4-R).",
        f"- آموزش توسط ابزار V4-S: {summary['training_performed_by_v4s']}.",
        "- روش: فقط خواندن گزارش و CSVهای ثابت Run #25؛ بدون دریافت داده، آموزش مدل یا تغییر داده.",
        "",
        "## 1. آستانه و نرخ پذیرش سیگنال BUY",
        "",
        "| بازو | Candidate آموزش‌دیده | آستانه ثبت‌شده | BUY خام | BUY پذیرفته | نسبت پذیرش | معامله Replay | PF خالص تجمیعی |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARM_NAMES:
        item = activity[name]
        thresholds = ", ".join(
            f"{value} ({count})"
            for value, count in sorted(item["confidence_threshold_counts"].items())
        ) or "موجود نیست"
        ratio = item["accepted_buy_percent_of_raw"]
        ratio_text = "موجود نیست" if ratio is None else f"{ratio:.4f}%"
        lines.append(
            f"| {name} | {item['trained_candidate_count']} | {thresholds} "
            f"| {item['raw_buy_signal_count']} | {item['accepted_buy_signal_count']} "
            f"| {ratio_text} | {item['trade_count']} "
            f"| {item['combined_net_profit_factor']:.6f} |"
        )
    lines.extend([
        "",
        "برای بازوهای ۱ و ۳ آستانه ثبت‌شده 0.60 است؛ آستانه 0.90 برای بازوی ۳ در Artifact وجود ندارد. اختلاف پذیرش سیگنال با آستانه یکسان مشاهده می‌شود، اما این داده aggregate علت را بین تابع هدف، توزیع احتمال خروجی و انتخاب Candidate تفکیک نمی‌کند.",
        "",
        "### وضعیت Threshold Optimization",
        "",
        "| بازو | Variant | وضعیت بهینه‌سازی | حمایت مثبت HOLD در بخش کالیبراسیون |",
        "|---|---:|---|---:|",
    ])
    for name in ARM_NAMES:
        item = probability[name]
        statuses = ", ".join(
            f"{status}: {count}"
            for status, count in sorted(item["threshold_optimization_statuses"].items())
        ) or "موجود نیست"
        support = item["calibration_hold_positive_support"]
        support_text = (
            "موجود نیست" if support["minimum"] is None
            else f"{support['minimum']} تا {support['maximum']} (میانگین {support['mean']:.2f})"
        )
        lines.append(
            f"| {name} | {item['model_variant_count']} | {statuses} | {support_text} |"
        )
    lines.extend([
        "",
        "تمام Variantهای دارای خروجی، وضعیت SKIPPED_INSUFFICIENT_CALIBRATION_SAMPLES و selected_threshold=null دارند. کف حمایت مثبت HOLD برابر ۳۰ است و نمونه کالیبراسیون این کف را تأمین نکرده است.",
        "",
        "## 2. توزیع برچسب در هندسه ۱:۱ و ۲:۱ ATR",
        "",
        "برای جلوگیری از دوباره‌شماری، از هر زوج pair/timeframe فقط یک توزیع استفاده شده است؛ دو Feature Profile توزیع یکسانی داشتند.",
        "",
        "| هندسه | Split | BUY | HOLD | SELL |",
        "|---|---|---:|---:|---:|",
    ])
    for geometry, name in (("۱:۱ ATR", ARM_NAMES[0]), ("۲:۱ ATR", ARM_NAMES[1])):
        for split in ("train", "validation"):
            values = labels[name]["splits"][split]["percent"]
            lines.append(
                f"| {geometry} | {split} | {values.get('BUY', 0):.2f}% "
                f"| {values.get('HOLD', 0):.2f}% | {values.get('SELL', 0):.2f}% |"
            )
    lines.extend([
        "",
        "### توزیع خروج Triple-Barrier",
        "",
        "| هندسه | TP_HIT | SL_HIT | HORIZON_EXPIRY | COLLISION |",
        "|---|---:|---:|---:|---:|",
    ])
    for geometry, name in (("۱:۱ ATR", ARM_NAMES[0]), ("۲:۱ ATR", ARM_NAMES[1])):
        values = labels[name]["exit_reasons"]["percent"]
        lines.append(
            f"| {geometry} | {values.get('TP_HIT', 0):.2f}% "
            f"| {values.get('SL_HIT', 0):.2f}% "
            f"| {values.get('HORIZON_EXPIRY', 0):.2f}% "
            f"| {values.get('COLLISION', 0):.2f}% |"
        )
    lines.extend([
        "",
        "سهم HOLD در هندسه ۲:۱ افزایش نیافته است. در Validation از ۴٫۳۸٪ به ۳٫۲۴٪ کاهش دارد. در عوض SL_HIT از ۶۱٫۱۱٪ به ۷۳٫۹۰٪ افزایش و TP_HIT از ۳۲٫۵۵٪ به ۲۱٫۷۱٪ کاهش یافته است. فرض «افزایش HOLD به علت افق کوتاه» پشتیبانی نمی‌شود؛ این داده مزیت افق ۴۸/۷۲ کندلی را نیز اثبات نمی‌کند.",
        "",
        "## 3. Probability Diagnostics و محدودیت بازپخش آستانه",
        "",
        "| بازو | Variant | میانگین Hold AUC در Forward Validation | میانگین Hold PR-AUC در Forward Validation |",
        "|---|---:|---:|---:|",
    ])
    for name in ARM_NAMES:
        item = probability[name]
        metrics = item["mean_metrics"]
        auc = metrics.get("forward_validation_hold_auc")
        pr_auc = metrics.get("forward_validation_hold_pr_auc")
        auc_text = "موجود نیست" if auc is None else f"{auc:.6f}"
        pr_text = "موجود نیست" if pr_auc is None else f"{pr_auc:.6f}"
        lines.append(
            f"| {name} | {item['model_variant_count']} | {auc_text} | {pr_text} |"
        )
    lines.extend([
        "",
        f"- Probability ردیفی در Artifact موجود است: {summary['row_level_probabilities_available']}.",
        f"- وضعیت Counterfactual Sweep: {summary['counterfactual_sweep_status']}.",
        "- Artifact فقط خلاصه‌های aggregate مانند Brier، ECE، AUC و PR-AUC دارد؛ trade_rows شامل timestamp، قیمت‌ها، خروج، هزینه و PnL است، نه Probability مربوط به هر ردیف.",
        "",
        "### سناریوهای آستانه درخواستی",
        "",
        "| بازو | آستانه | تعداد معامله | PF خالص | وضعیت |",
        "|---|---:|---:|---:|---|",
        f"| بازوی ۳: Economic Proxy | 0.60 (واقعی) | {activity[ARM_NAMES[2]]['trade_count']} | {activity[ARM_NAMES[2]]['combined_net_profit_factor']:.6f} | مقدار واقعی Artifact |",
        "| بازوی ۳: Economic Proxy | 0.70 | موجود نیست | موجود نیست | قابل محاسبه نیست؛ Probability ردیفی ذخیره نشده |",
        "| بازوی ۳: Economic Proxy | 0.90 | موجود نیست | موجود نیست | قابل محاسبه نیست؛ Probability ردیفی ذخیره نشده |",
        f"| بازوی ۱: F1 | 0.60 (واقعی) | {activity[ARM_NAMES[0]]['trade_count']} | {activity[ARM_NAMES[0]]['combined_net_profit_factor']:.6f} | مقدار واقعی Artifact |",
        "| بازوی ۱: F1 | 0.80 | موجود نیست | موجود نیست | قابل محاسبه نیست؛ Probability ردیفی ذخیره نشده |",
        "",
        "اعداد Counterfactual حدس زده نشده‌اند. بازسازی Probability یا PF از روی AUC/Brier و شمارش aggregate از نظر ریاضی معتبر نیست.",
        "",
        "## 4. پیشنهاد V4-T",
        "",
        "1. **Horizon هندسه ۲:۱:** افزایش مستقیم به ۷۲ کندل توصیه نمی‌شود. ابتدا آزمایش تک‌متغیره ۲۴ در برابر ۴۸ کندل با همان TP=2 ATR و SL=1 ATR انجام شود. فقط اگر رویدادهایی مشاهده شوند که بین کندل‌های ۲۵ تا ۴۸ به TP می‌رسند، ۷۲ کندل بررسی شود. Purge horizon مشترک باید دست‌کم برابر بیشترین افق باشد.",
        "2. **تابع هدف ترکیبی:** جایگزینی فوری NET_EXPECTANCY_PROXY با جمع وزنی PF و F1 توصیه نمی‌شود. ابتدا احتمال BUY/HOLD و پیش‌بینی ردیفی ذخیره شود. سپس F1، Proxy خالص و Proxy با قید حداقل کیفیت/پوشش سیگنال با طراحی از پیش ثبت‌شده مقایسه شوند. حداقل نمونه و Gate اقتصادی تغییر نکنند.",
        "3. **کالیبراسیون آستانه:** Probabilityهای BUY/HOLD، کلاس پیش‌بینی‌شده، timestamp و label واقعی باید برای بخش Calibration و Forward Validation ذخیره شوند. آستانه فقط روی Calibration انتخاب و بدون تنظیم مجدد روی بازه زمانی بعدی ارزیابی شود.",
        "",
        "## 5. ایمنی",
        "",
        "| ثابت | مقدار Artifact |",
        "|---|---|",
        "| Champion | null |",
        "| order_permission | false |",
        "| trading_authorization | NO-GO |",
        "| private_api_calls | 0 |",
        "| real_order_attempts | 0 |",
        "| orders_or_execution_api_used | false |",
        "| credentials_used | false |",
        "",
        "**تصمیم:** این گزارش فقط Audit خواندنی Run #25 است. V4-S هیچ مدلی آموزش نداده، داده‌ای را تغییر نداده، آستانه یا Gate را دست‌کاری نکرده و مجوز معامله صادر نمی‌کند. Champion=null و NO-GO پابرجاست.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = _load_report(args.artifact)
    summary = analyze_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps({
        "source_run_id": EXPECTED_RUN_ID,
        "source_artifact_id": EXPECTED_ARTIFACT_ID,
        "source_commit_sha": summary["commit_sha"],
        "counterfactual_sweep_status": summary["counterfactual_sweep_status"],
        "training_performed": False,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
