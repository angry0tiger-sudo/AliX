#!/usr/bin/env python3
"""Run an offline, synthetic-only Phase 9 integration chain."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import tempfile

from external_data.contracts import (
    NLP_RESULT_SCHEMA_VERSION,
    NewsFeature,
    ProviderError,
)
from external_data.point_in_time import join_news_features_as_of
from risk.allocator import DynamicRiskAllocator
from risk.contracts import (
    AccountSnapshot,
    ActionType,
    MarketRules,
    MarketSnapshot,
    ModelRiskProposal,
    RiskPolicy,
    Side,
    stable_fingerprint,
)
from risk.gate import RiskGate
from risk.kill_switch import KillSwitchState
from services.data_quality.gate import (
    EXTERNAL_RECORD_SCHEMA_VERSION,
    MODEL_OUTPUT_SCHEMA_VERSION,
    QualityResult,
    QualityStatus,
    validate_candles,
    validate_external_record,
    validate_model_output,
)
from services.reporting.engine import DailyReport, ReportingEngine
from services.simulation.ledger import (
    LedgerSnapshot,
    SQLiteDecisionLedger,
    SimulationDecisionRecord,
    SimulationStatus,
)

D = Decimal
NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    model_action: str = "ENTRY"
    model_available: bool = True
    stale_candles: bool = False
    candle_gap: bool = False
    stale_news: bool = False
    conflicting_news: bool = False
    future_external_record: bool = False
    atr: Decimal = D("2")
    kill_switch_latched: bool = False
    outcome: str = "normal"


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("baseline_entry"),
    Scenario("stale_candles", stale_candles=True),
    Scenario("candle_gap", candle_gap=True),
    Scenario("stale_news", stale_news=True),
    Scenario("conflicting_news", conflicting_news=True),
    Scenario("future_external_availability", future_external_record=True),
    Scenario("model_unavailable", model_available=False),
    Scenario("model_hold", model_action="HOLD"),
    Scenario("atr_spike", atr=D("20")),
    Scenario("kill_switch_latched", kill_switch_latched=True),
    Scenario("synthetic_order_timeout", outcome="unknown"),
)


def _policy() -> RiskPolicy:
    return RiskPolicy(
        revision="phase9-synthetic-policy-v1",
        owner_approved=True,
        absolute_daily_loss_cap=D("100"),
        absolute_gross_exposure_cap=D("10000"),
        daily_risk_budget_fraction=D("0.05"),
        per_trade_risk_fraction=D("0.01"),
        max_stake_fraction_of_free_balance=D("0.5"),
        max_gross_exposure_fraction=D("1"),
        max_correlated_exposure_fraction=D("0.8"),
        confidence_floor=D("0.3"),
        max_uncertainty=D("0.8"),
        max_calibration_error=D("0.4"),
        reference_atr_fraction=D("0.02"),
        max_atr_fraction=D("0.1"),
        stop_atr_multiple=D("2"),
        fee_rate=D("0.001"),
        slippage_rate=D("0.001"),
        minimum_edge_after_costs=D("0.0001"),
        max_market_age_seconds=180,
        max_account_age_seconds=180,
        regime_multipliers=(("low", D("1")), ("high", D("0.6"))),
    )


def _account() -> AccountSnapshot:
    return AccountSnapshot(
        as_of=NOW,
        currency="USDT",
        equity=D("10000"),
        free_balance=D("9000"),
        realized_pnl=D("0"),
        unrealized_pnl=D("0"),
        fees_paid=D("0"),
        open_position_risk=D("0"),
        open_order_risk=D("0"),
        open_order_notional=D("0"),
        gross_exposure=D("0"),
        correlated_exposure=D("0"),
        reconciled=True,
        orders_reconciled=True,
        source_snapshot_hash=hashlib.sha256(b"phase9-synthetic-account").hexdigest(),
    )


def _market(scenario: Scenario) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT",
        as_of=NOW,
        price=D("100"),
        atr=scenario.atr,
        regime="low",
        source_revision="phase9-synthetic-market-v1",
    )


def _rules() -> MarketRules:
    return MarketRules(
        symbol="BTC/USDT",
        revision="phase9-synthetic-rules-v1",
        approved=True,
        price_tick=D("0.01"),
        quantity_step=D("0.001"),
        min_quantity=D("0.001"),
        max_quantity=D("1000"),
        min_notional=D("10"),
        max_notional=D("100000"),
    )


def _candle_rows(scenario: Scenario) -> list[dict[str, object]]:
    if scenario.stale_candles:
        latest = NOW - timedelta(minutes=20)
    else:
        latest = NOW - timedelta(minutes=1)
    previous = (
        latest - timedelta(minutes=3)
        if scenario.candle_gap
        else latest - timedelta(minutes=1)
    )
    return [
        {
            "date": previous.timestamp(),
            "open": 100.0,
            "high": 102.0,
            "low": 98.0,
            "close": 100.0,
            "volume": 10.0,
        },
        {
            "date": latest.timestamp(),
            "open": 100.0,
            "high": 102.0,
            "low": 98.0,
            "close": 100.0,
            "volume": 10.0,
        },
    ]


def _external_record(scenario: Scenario) -> dict[str, object]:
    available_at = (
        NOW + timedelta(minutes=1)
        if scenario.future_external_record
        else NOW - timedelta(minutes=3)
    )
    return {
        "schema_version": EXTERNAL_RECORD_SCHEMA_VERSION,
        "provider_id": "phase9.synthetic.market",
        "source_id": "fixture-market-feed",
        "event_id": f"external-{scenario.scenario_id}",
        "event_timestamp": (NOW - timedelta(minutes=10)).isoformat(),
        "ingested_at": (available_at - timedelta(minutes=2)).isoformat(),
        "available_at": available_at.isoformat(),
        "unit": "count",
        "quote_currency": "USDT",
        "revision": "synthetic-r1",
        "license_status": "approved",
        "rate_limit_status": "ok",
    }


def _news_feature(
    *,
    event_id: str,
    available_at: datetime,
    revision: str,
    content: str,
) -> NewsFeature:
    published = available_at - timedelta(minutes=10)
    ingested = available_at - timedelta(minutes=5)
    processed = available_at - timedelta(minutes=2)
    return NewsFeature(
        event_id=event_id,
        provider_id="phase9.synthetic.news",
        source_id="fixture-news-feed",
        published_at=published,
        ingested_at=ingested,
        available_at=available_at,
        processed_at=processed,
        revision=revision,
        asset_mapping=("BTC/USDT",),
        license_status="approved",
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        sentiment_label="neutral",
        sentiment_score=0.0,
        confidence=0.8,
        event_type="synthetic",
        model_id="local-fixture-model",
        model_revision="synthetic-nlp-v1",
        calibration_status="calibrated",
        schema_version=NLP_RESULT_SCHEMA_VERSION,
    )


def _join_news(scenario: Scenario) -> tuple[int, str | None]:
    available_at = (
        NOW - timedelta(hours=2) if scenario.stale_news else NOW - timedelta(minutes=3)
    )
    first = _news_feature(
        event_id=f"news-{scenario.scenario_id}",
        available_at=available_at,
        revision="synthetic-r1",
        content="synthetic market context one",
    )
    features = [first]
    if scenario.conflicting_news:
        features.append(
            _news_feature(
                event_id=first.event_id,
                available_at=available_at,
                revision="synthetic-r2",
                content="synthetic market context conflict",
            )
        )
    try:
        joined = join_news_features_as_of(
            features,
            asset="BTC/USDT",
            decision_timestamp=NOW,
            window_start=NOW - timedelta(hours=1),
        )
    except ProviderError as exc:
        return 0, str(exc).split(":", maxsplit=1)[0]
    return len(joined.records), None


def _model_output(scenario: Scenario) -> dict[str, object]:
    return {
        "schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
        "produced_at": NOW.isoformat(),
        "model_revision": "phase9-synthetic-model-v1",
        "proposal": scenario.model_action,
        "confidence": 0.8,
        "uncertainty": 0.1,
    }


def _proposal(scenario: Scenario) -> ModelRiskProposal:
    return ModelRiskProposal(
        decision_id=scenario.scenario_id,
        model_revision="phase9-synthetic-model-v1",
        side=Side.LONG,
        regime="low",
        atr_snapshot=scenario.atr,
        calibrated_confidence=D("0.8"),
        uncertainty=D("0.1"),
        calibration_error=D("0.1"),
        expected_edge_after_costs=D("0.01"),
    )


def _candle_result(scenario: Scenario) -> QualityResult:
    return validate_candles(
        _candle_rows(scenario),
        max_age_seconds=180,
        now_epoch=NOW.timestamp(),
        expected_interval_seconds=60,
        gap_policy="reject",
    )


def _risk_inputs(
    scenario: Scenario,
) -> tuple[RiskPolicy, AccountSnapshot, MarketSnapshot, MarketRules]:
    return _policy(), _account(), _market(scenario), _rules()


def _record_rejection(
    scenario: Scenario,
    ledger: SQLiteDecisionLedger,
    fingerprint: str,
    reasons: list[str],
    status: SimulationStatus = SimulationStatus.REJECTED,
    *,
    model_revision: str | None = "phase9-synthetic-model-v1",
    trace: dict[str, object] | None = None,
) -> dict[str, object]:
    safe_reasons = list(dict.fromkeys(reasons)) or ["entry_not_authorized"]
    trace_payload = dict(trace or {})
    trace_payload.setdefault("final_status", status.value)
    ledger.append(
        SimulationDecisionRecord(
            decision_id=scenario.scenario_id,
            decided_at=NOW,
            input_fingerprint=fingerprint,
            model_revision=model_revision,
            risk_approved=False,
            reasons=tuple(safe_reasons),
            simulation_status=status,
            trace_json=json.dumps(trace_payload, sort_keys=True, separators=(",", ":")),
        )
    )
    return {
        "scenario_id": scenario.scenario_id,
        "status": status.value,
        "risk_approved": False,
        "order_permission": False,
        "reasons": safe_reasons,
        "simulated_fill": False,
        "exchange_calls": 0,
    }


def _validate_input_stages(
    scenario: Scenario,
) -> tuple[list[str], int, dict[str, object] | None, dict[str, object]]:
    reasons: list[str] = []
    candles = _candle_result(scenario)
    external_source = _external_record(scenario)
    if not candles.passed or candles.status is not QualityStatus.VALID:
        reasons.extend(code.value for code in candles.reason_codes)
    external = validate_external_record(
        external_source,
        as_of_timestamp=NOW.isoformat(),
        expected_schema_version=EXTERNAL_RECORD_SCHEMA_VERSION,
    )
    if not external.passed or external.status is not QualityStatus.VALID:
        reasons.extend(code.value for code in external.reason_codes)
    news_count, news_error = _join_news(scenario)
    if news_error:
        reasons.append(news_error)
    elif news_count == 0:
        reasons.append("news_stale_or_unavailable")
    model_check: dict[str, object] | None = None
    model_result: dict[str, object] = {
        "available": scenario.model_available,
        "output": None,
        "status": "NOT_RUN",
        "reason_codes": [],
    }
    if not scenario.model_available:
        reasons.append("model_unavailable")
    else:
        model_check = _model_output(scenario)
        checked_model = validate_model_output(
            model_check,
            decision_timestamp=NOW.isoformat(),
        )
        model_result = {
            "available": True,
            "output": model_check,
            "status": (
                checked_model.status.value
                if checked_model.status is not None
                else "UNKNOWN"
            ),
            "passed": checked_model.passed,
            "reason_codes": [code.value for code in checked_model.reason_codes],
            "fingerprint": checked_model.fingerprint,
        }
        if not checked_model.passed or checked_model.status is not QualityStatus.VALID:
            reasons.extend(code.value for code in checked_model.reason_codes)
    trace: dict[str, object] = {
        "decision_timestamp": NOW.isoformat(),
        "input_fingerprint_schema": "alix.phase9.input.v1",
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "model_action": scenario.model_action,
            "model_available": scenario.model_available,
            "stale_candles": scenario.stale_candles,
            "candle_gap": scenario.candle_gap,
            "stale_news": scenario.stale_news,
            "conflicting_news": scenario.conflicting_news,
            "future_external_record": scenario.future_external_record,
            "atr": str(scenario.atr),
            "kill_switch_latched": scenario.kill_switch_latched,
            "outcome": scenario.outcome,
        },
        "inputs": {
            "candles": _candle_rows(scenario),
            "external_record": external_source,
        },
        "data_quality": {
            "candles": {
                "status": (
                    candles.status.value if candles.status is not None else "UNKNOWN"
                ),
                "reason_codes": [code.value for code in candles.reason_codes],
                "fingerprint": candles.fingerprint,
            },
            "external_record": {
                "status": (
                    external.status.value if external.status is not None else "UNKNOWN"
                ),
                "reason_codes": [code.value for code in external.reason_codes],
                "fingerprint": external.fingerprint,
            },
        },
        "point_in_time_news": {
            "selected_record_count": news_count,
            "error_code": news_error,
            "window_start": (NOW - timedelta(hours=1)).isoformat(),
            "decision_timestamp": NOW.isoformat(),
        },
        "model": model_result,
    }
    return list(dict.fromkeys(reasons)), news_count, model_check, trace


def _evaluate_scenario(
    scenario: Scenario,
    ledger: SQLiteDecisionLedger,
) -> dict[str, object]:
    fingerprint = stable_fingerprint(
        {
            "scenario": scenario,
            "decision_timestamp": NOW,
            "source": "synthetic-only",
        }
    )
    input_reasons, news_count, model, trace = _validate_input_stages(scenario)
    trace["input_fingerprint"] = fingerprint
    if input_reasons:
        return _record_rejection(
            scenario,
            ledger,
            fingerprint,
            input_reasons,
            model_revision=(
                "phase9-synthetic-model-v1" if scenario.model_available else None
            ),
            trace=trace,
        )
    if not scenario.model_available or model is None:
        return _record_rejection(
            scenario,
            ledger,
            fingerprint,
            ["model_unavailable"],
            model_revision=None,
            trace=trace,
        )
    if scenario.model_action != "ENTRY":
        return _record_rejection(
            scenario,
            ledger,
            fingerprint,
            ["model_proposal_not_entry"],
            SimulationStatus.NOT_SIMULATED,
            trace=trace,
        )

    policy, account, market, rules = _risk_inputs(scenario)
    allocation_result = DynamicRiskAllocator().allocate(
        policy=policy,
        account=account,
        market=market,
        rules=rules,
        proposal=_proposal(scenario),
        now=NOW,
    )
    kill_state = (
        KillSwitchState(True, "synthetic_fault_injection", NOW, 1, NOW)
        if scenario.kill_switch_latched
        else KillSwitchState(False, "", None, 1, NOW)
    )
    gate = RiskGate().evaluate_entry(
        allocation=allocation_result.allocation,
        account=account,
        market=market,
        rules=rules,
        policy=policy,
        kill_switch=kill_state,
        evaluated_at=NOW,
    )
    if gate.order_permission is not False:
        raise AssertionError("risk_gate_order_permission_must_remain_false")

    trace["allocator"] = {
        "approved": allocation_result.approved,
        "reasons": list(allocation_result.reasons),
        "daily_risk_budget": str(allocation_result.daily_risk_budget),
        "trade_risk_budget": str(allocation_result.trade_risk_budget),
        "fingerprint": allocation_result.fingerprint,
        "allocation": (
            {
                "decision_id": allocation_result.allocation.decision_id,
                "model_revision": allocation_result.allocation.model_revision,
                "policy_revision": allocation_result.allocation.policy_revision,
                "account_snapshot_hash": allocation_result.allocation.account_snapshot_hash,
                "market_revision": allocation_result.allocation.market_revision,
                "rules_revision": allocation_result.allocation.rules_revision,
                "quantity": str(allocation_result.allocation.quantity),
                "stake": str(allocation_result.allocation.stake),
                "entry_price": str(allocation_result.allocation.entry_price),
                "stop_price": str(allocation_result.allocation.stop_price),
                "estimated_loss_including_costs": str(
                    allocation_result.allocation.estimated_loss_including_costs
                ),
                "fingerprint": allocation_result.allocation.fingerprint,
            }
            if allocation_result.allocation is not None
            else None
        ),
    }
    trace["risk_gate"] = {
        "decision_id": gate.decision_id,
        "risk_approved": gate.risk_approved,
        "risk_reduction_allowed": gate.risk_reduction_allowed,
        "order_permission": False,
        "reasons": list(gate.reasons),
        "estimated_loss_including_costs": str(gate.estimated_loss_including_costs),
        "fingerprint": gate.fingerprint,
    }
    risk_reasons = list(allocation_result.reasons)
    risk_reasons.extend(gate.reasons)
    if (
        not allocation_result.approved
        or allocation_result.allocation is None
        or not gate.risk_approved
    ):
        trace["final_status"] = SimulationStatus.REJECTED.value
        result = _record_rejection(
            scenario, ledger, fingerprint, risk_reasons, trace=trace
        )
        result["risk_gate_fingerprint"] = gate.fingerprint
        result["allocator_fingerprint"] = allocation_result.fingerprint
        result["news_records_selected"] = news_count
        result["trace_reconstructable"] = True
        return result

    allocation = allocation_result.allocation
    if scenario.outcome == "unknown":
        ledger.append(
            SimulationDecisionRecord(
                decision_id=scenario.scenario_id,
                decided_at=NOW,
                input_fingerprint=fingerprint,
                model_revision=allocation.model_revision,
                risk_approved=True,
                reasons=("synthetic_unknown_outcome_no_retry",),
                simulation_status=SimulationStatus.SYNTHETIC_UNKNOWN_NO_RETRY,
                trace_json=json.dumps(
                    trace
                    | {
                        "final_status": SimulationStatus.SYNTHETIC_UNKNOWN_NO_RETRY.value,
                        "timeout_policy": "NO_RETRY",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
        return {
            "scenario_id": scenario.scenario_id,
            "status": SimulationStatus.SYNTHETIC_UNKNOWN_NO_RETRY.value,
            "risk_approved": True,
            "order_permission": False,
            "reasons": ["synthetic_unknown_outcome_no_retry"],
            "simulated_fill": False,
            "retry_count": 0,
            "exchange_calls": 0,
            "risk_gate_fingerprint": gate.fingerprint,
        }

    entry = market.price
    exit_price = allocation.stop_price
    gross_pnl = (exit_price - entry) * allocation.quantity
    costs = (
        (entry + exit_price)
        * allocation.quantity
        * (policy.fee_rate + policy.slippage_rate)
    )
    net_pnl = gross_pnl - costs
    ledger.append(
        SimulationDecisionRecord(
            decision_id=scenario.scenario_id,
            decided_at=NOW,
            input_fingerprint=fingerprint,
            model_revision=allocation.model_revision,
            risk_approved=True,
            reasons=(),
            simulation_status=SimulationStatus.SIMULATED,
            simulated_quantity=allocation.quantity,
            simulated_entry_price=entry,
            simulated_stop_price=allocation.stop_price,
            simulated_exit_price=exit_price,
            simulated_net_pnl=net_pnl,
            trace_json=json.dumps(
                trace
                | {
                    "final_status": SimulationStatus.SIMULATED.value,
                    "synthetic_fill": {
                        "quantity": str(allocation.quantity),
                        "entry_price": str(entry),
                        "stop_price": str(allocation.stop_price),
                        "exit_price": str(exit_price),
                        "net_pnl": str(net_pnl),
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    )
    return {
        "scenario_id": scenario.scenario_id,
        "status": SimulationStatus.SIMULATED.value,
        "risk_approved": True,
        "order_permission": False,
        "reasons": [],
        "simulated_fill": True,
        "simulated_quantity": str(allocation.quantity),
        "simulated_entry_price": str(entry),
        "simulated_stop_price": str(allocation.stop_price),
        "simulated_exit_price": str(exit_price),
        "simulated_net_pnl": str(net_pnl),
        "exchange_calls": 0,
        "risk_gate_fingerprint": gate.fingerprint,
        "allocator_fingerprint": allocation_result.fingerprint,
        "news_records_selected": news_count,
    }


def _exercise_risk_reduction() -> dict[str, object]:
    result = RiskGate().evaluate_reduction(
        ActionType.EXIT,
        "phase9-synthetic-risk-reduction",
        NOW,
    )
    return {
        "decision_id": result.decision_id,
        "risk_reduction_allowed": result.risk_reduction_allowed,
        "order_permission": result.order_permission,
        "reasons": list(result.reasons),
    }


def _render_mock_report(snapshot: LedgerSnapshot) -> dict[str, object]:
    delivered: list[tuple[str, str]] = []

    def fake_sender(recipient: str, message: str) -> bool:
        delivered.append((recipient, message))
        return True

    daily = DailyReport(
        date=NOW.date().isoformat(),
        summary=(
            "SYNTHETIC E2E ONLY; reconciliation NOT_RECONCILED; "
            "trading authorization NO-GO"
        ),
        metrics={
            "rows": float(snapshot.row_count),
            "simulated": float(snapshot.simulated_count),
            "rejected": float(snapshot.rejected_count),
            "unknown_no_retry": float(snapshot.unknown_no_retry_count),
            "simulated_net_pnl": float(snapshot.simulated_net_pnl),
        },
    )
    engine = ReportingEngine(default_chat_id="10001")
    rendered = engine.render_text(daily)
    delivered_ok = engine.deliver(
        daily,
        enabled=True,
        chat_id="10001",
        allowlist={"10001"},
        sender=fake_sender,
    )
    return {
        "rendered": rendered,
        "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "mock_delivery_success": delivered_ok,
        "mock_delivery_calls": len(delivered),
        "delivery_transport": "injected_fake_sender_no_network",
        "source_status": snapshot.source_status,
        "reconciliation_status": snapshot.reconciliation_status,
        "order_permission": False,
    }


def run_e2e(commit_sha: str, output_path: Path | None = None) -> dict[str, object]:
    if not SHA_PATTERN.fullmatch(commit_sha):
        raise ValueError("full_commit_sha_required")
    with tempfile.TemporaryDirectory(prefix="alix-phase9-synthetic-") as temp_dir:
        ledger = SQLiteDecisionLedger(Path(temp_dir) / "decision-ledger.sqlite")
        results = [_evaluate_scenario(scenario, ledger) for scenario in SCENARIOS]
        baseline = next(
            item for item in results if item["scenario_id"] == "baseline_entry"
        )
        if baseline["status"] != SimulationStatus.SIMULATED.value:
            raise AssertionError(f"baseline_scenario_failed:{baseline['status']}")
        expected = {
            "baseline_entry": SimulationStatus.SIMULATED.value,
            "stale_candles": SimulationStatus.REJECTED.value,
            "candle_gap": SimulationStatus.REJECTED.value,
            "stale_news": SimulationStatus.REJECTED.value,
            "conflicting_news": SimulationStatus.REJECTED.value,
            "future_external_availability": SimulationStatus.REJECTED.value,
            "model_unavailable": SimulationStatus.REJECTED.value,
            "model_hold": SimulationStatus.NOT_SIMULATED.value,
            "atr_spike": SimulationStatus.REJECTED.value,
            "kill_switch_latched": SimulationStatus.REJECTED.value,
            "synthetic_order_timeout": SimulationStatus.SYNTHETIC_UNKNOWN_NO_RETRY.value,
        }
        actual = {str(item["scenario_id"]): str(item["status"]) for item in results}
        if actual != expected:
            raise AssertionError(f"phase9_scenario_matrix_mismatch:{actual!r}")
        snap = ledger.snapshot(data_cutoff=NOW, generated_at=NOW)
        baseline_record = ledger.get_record("baseline_entry")
        if baseline_record is None:
            raise AssertionError("phase9_ledger_baseline_record_missing")
        baseline_trace = baseline_record.get("trace")
        if not isinstance(baseline_trace, dict):
            raise AssertionError("phase9_decision_trace_not_reconstructable")
        if not all(
            key in baseline_trace
            for key in ("inputs", "model", "allocator", "risk_gate", "synthetic_fill")
        ):
            raise AssertionError("phase9_decision_trace_missing_required_stage")
        if (
            snap.order_permission is not False
            or snap.reconciliation_status != "NOT_RECONCILED"
        ):
            raise AssertionError(
                "synthetic_ledger_must_not_claim_order_permission_or_reconciliation"
            )

        baseline_fingerprint = stable_fingerprint({"baseline": "idempotent-replay"})
        replay = SimulationDecisionRecord(
            decision_id="phase9-replay-check",
            decided_at=NOW,
            input_fingerprint=baseline_fingerprint,
            model_revision="phase9-synthetic-model-v1",
            risk_approved=False,
            reasons=("idempotency_probe",),
            simulation_status=SimulationStatus.REJECTED,
        )
        first_append = ledger.append(replay)
        second_append = ledger.append(replay)
        if not first_append or second_append:
            raise AssertionError("simulation_ledger_idempotency_failed")
        snapshot_after_replay = ledger.snapshot(data_cutoff=NOW, generated_at=NOW)
        if snapshot_after_replay.row_count != snap.row_count + 1:
            raise AssertionError("simulation_ledger_replay_changed_record_count")
        reduction = _exercise_risk_reduction()
        if reduction["order_permission"] is not False:
            raise AssertionError("risk_reduction_must_not_authorize_order")
        report = {
            "schema_version": "alix.phase9.synthetic-e2e.v1",
            "commit_sha": commit_sha,
            "evaluation_status": "SYNTHETIC_E2E_ONLY",
            "dataset_kind": "synthetic",
            "scenario_count": len(results),
            "scenario_results": results,
            "risk_reduction_contract": reduction,
            "ledger": {
                "report_id": snap.report_id,
                "data_cutoff": snap.data_cutoff.isoformat(),
                "generated_at": snap.generated_at.isoformat(),
                "snapshot_hash": snap.snapshot_hash,
                "row_count": snap.row_count,
                "simulated_count": snap.simulated_count,
                "rejected_count": snap.rejected_count,
                "not_simulated_count": snap.not_simulated_count,
                "unknown_no_retry_count": snap.unknown_no_retry_count,
                "simulated_net_pnl": str(snap.simulated_net_pnl),
                "source_status": snap.source_status,
                "reconciliation_status": snap.reconciliation_status,
                "order_permission": False,
            },
            "idempotency": {
                "first_append": first_append,
                "identical_replay_inserted": second_append,
                "row_count_before_probe": snap.row_count,
                "row_count_after_probe": snapshot_after_replay.row_count,
                "probe_changes_snapshot": snapshot_after_replay.snapshot_hash
                != snap.snapshot_hash,
            },
            "reporting": _render_mock_report(snap),
            "safety": {
                "order_permission": False,
                "real_order_attempts": 0,
                "exchange_calls": 0,
                "live_or_paper_sessions_started": 0,
                "real_credentials_used": False,
                "champion": None,
                "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
                "economic_acceptance": "NO_GO",
                "trading_authorization": "NO_GO",
            },
            "limitations": [
                "Decision ledger is a synthetic simulation ledger, not the actual Freqtrade trade database.",
                "No source-of-truth exchange reconciliation is performed; PnL is synthetic and labelled NOT_RECONCILED.",
                "The simulated timeout tests no-retry mechanics and does not represent an exchange request.",
                "Capacity, production filesystem isolation, and owner-host Windows/PostgreSQL integration are not measured by this report.",
            ],
        }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_e2e(args.commit_sha, args.output)
    print(
        json.dumps(
            {
                "commit_sha": report["commit_sha"],
                "evaluation_status": report["evaluation_status"],
                "scenario_count": report["scenario_count"],
                "ledger": report["ledger"],
                "safety": report["safety"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
