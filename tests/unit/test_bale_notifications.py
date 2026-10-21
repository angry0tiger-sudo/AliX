from __future__ import annotations

import asyncio
from typing import Any

from services.notifications.bale import (
    DEFAULT_BALE_CHAT_ID,
    BaleIOTransport,
    BaleNotificationService,
    BaleSettings,
)
from services.reporting.engine import DailyReport, ReportingEngine


class FakeTransport:
    def __init__(self, outcome: bool = True) -> None:
        self.outcome = outcome
        self.messages: list[tuple[str, str]] = []

    async def send_message(self, chat_id: str, text: str) -> bool:
        self.messages.append((chat_id, text))
        return self.outcome


class FakeBaleBot:
    def __init__(self) -> None:
        self.messages: list[tuple[int | str, str]] = []
        self.closed = False

    async def send_message(self, chat_id: int | str, text: str) -> object:
        self.messages.append((chat_id, text))
        return object()

    async def close(self) -> None:
        self.closed = True


def settings(
    *,
    enabled: bool = True,
    token: str = "mock-token",
    chat_id: str = DEFAULT_BALE_CHAT_ID,
    allowlist: frozenset[str] = frozenset({DEFAULT_BALE_CHAT_ID}),
) -> BaleSettings:
    return BaleSettings(enabled, token, chat_id, allowlist)


def test_default_chat_id_and_environment_override(monkeypatch: Any) -> None:
    monkeypatch.delenv("BALE_CHAT_ID", raising=False)
    assert BaleSettings.from_environment().chat_id == "832112407"
    monkeypatch.setenv("BALE_CHAT_ID", "123456")
    monkeypatch.setenv("ALIx_BALE_CHAT_ALLOWLIST", "123456")
    assert BaleSettings.from_environment().chat_id == "123456"


def test_bale_mock_transport_sends_to_configured_allowlisted_chat() -> None:
    transport = FakeTransport()
    service = BaleNotificationService(settings(), transport)

    result = asyncio.run(service.send_text("AliX phase one mock report"))

    assert result.sent
    assert result.reason == "sent"
    assert result.chat_id == DEFAULT_BALE_CHAT_ID
    assert transport.messages == [(DEFAULT_BALE_CHAT_ID, "AliX phase one mock report")]


def test_bale_transport_respects_disable_token_allowlist_and_message_gates() -> None:
    transport = FakeTransport()
    disabled = BaleNotificationService(settings(enabled=False), transport)
    no_token = BaleNotificationService(settings(token=""), transport)
    not_allowlisted = BaleNotificationService(
        settings(allowlist=frozenset({"777777"})), transport
    )
    invalid_message = BaleNotificationService(settings(), transport)

    assert asyncio.run(disabled.send_text("x")).reason == "reporting_disabled"
    assert asyncio.run(no_token.send_text("x")).reason == "missing_bot_token"
    assert (
        asyncio.run(not_allowlisted.send_text("x")).reason
        == "recipient_not_allowlisted"
    )
    assert asyncio.run(invalid_message.send_text(" ")).reason == "invalid_message"
    assert (
        asyncio.run(invalid_message.send_text("x" * 4097)).reason == "invalid_message"
    )
    assert len(transport.messages) == 0


def test_bale_transport_uses_injected_bot_factory_and_closes_client() -> None:
    bot = FakeBaleBot()
    transport = BaleIOTransport("mock-token", bot_factory=lambda token: bot)

    sent = asyncio.run(transport.send_message(DEFAULT_BALE_CHAT_ID, "mock-only"))

    assert sent
    assert bot.messages == [(832112407, "mock-only")]
    assert bot.closed


def test_reporting_engine_uses_changeable_default_chat_and_allowlist() -> None:
    engine = ReportingEngine(default_chat_id="chat-two")
    report = DailyReport("2026-10-21", "mock-only", {"trades": 0.0})
    received: list[tuple[str, str]] = []

    def sender(chat_id: str, text: str) -> bool:
        received.append((chat_id, text))
        return True

    assert engine.deliver(
        report,
        enabled=True,
        allowlist={"chat-two"},
        sender=sender,
    )
    assert received[0][0] == "chat-two"
    assert not engine.deliver(report, enabled=True, allowlist=set(), sender=sender)


def test_reporting_engine_default_bale_delivery_stays_disabled_without_env(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("ALIx_REPORTING_ENABLED", "false")
    monkeypatch.delenv("BALE_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ALIx_BALE_CHAT_ALLOWLIST", raising=False)
    engine = ReportingEngine()
    report = DailyReport("2026-10-21", "mock-only", {"trades": 0.0})

    result = asyncio.run(engine.deliver_bale(report))

    assert not result.sent
    assert result.reason == "reporting_disabled"
    assert result.chat_id == DEFAULT_BALE_CHAT_ID


def test_bale_settings_reject_invalid_chat_and_allowlist_ids() -> None:
    import pytest

    with pytest.raises(ValueError, match="chat_id must be numeric"):
        BaleSettings(enabled=True, chat_id="not-a-number")
    with pytest.raises(ValueError, match="allowlist entries must be numeric"):
        BaleSettings(
            enabled=True,
            chat_id=DEFAULT_BALE_CHAT_ID,
            allowlist=frozenset({"recipient-name"}),
        )


def test_bale_sender_handles_unavailable_rejected_and_raising_transport() -> None:
    class RaisingTransport:
        async def send_message(self, chat_id: str, text: str) -> bool:
            raise RuntimeError("mock transport failure")

    no_transport = BaleNotificationService(settings(), None)
    result_unavailable = asyncio.run(no_transport.send_text("mock"))
    assert result_unavailable.reason == "transport_unavailable"

    rejected = BaleNotificationService(settings(), FakeTransport(outcome=False))
    result_rejected = asyncio.run(rejected.send_text("mock"))
    assert not result_rejected.sent
    assert result_rejected.reason == "transport_rejected"

    raising = BaleNotificationService(settings(), RaisingTransport())
    result_failed = asyncio.run(raising.send_text("mock"))
    assert not result_failed.sent
    assert result_failed.reason == "transport_failed"


def test_bale_from_environment_builds_transport_without_sending(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("ALIx_REPORTING_ENABLED", "true")
    monkeypatch.setenv("BALE_BOT_TOKEN", "mock-token")
    monkeypatch.setenv("BALE_CHAT_ID", DEFAULT_BALE_CHAT_ID)
    monkeypatch.setenv("ALIx_BALE_CHAT_ALLOWLIST", "")
    service = BaleNotificationService.from_environment()
    result = asyncio.run(
        service.send_text("no delivery because recipient is not allowlisted")
    )
    assert result.reason == "recipient_not_allowlisted"
