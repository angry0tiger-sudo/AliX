"""Bale notification boundary with disabled-by-default, fail-closed delivery."""

from __future__ import annotations

import importlib
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast


DEFAULT_BALE_CHAT_ID = "832112407"
MAX_MESSAGE_LENGTH = 4096


class BaleBotProtocol(Protocol):
    async def send_message(self, chat_id: int | str, text: str) -> Any:
        ...

    async def close(self) -> None:
        ...


BotFactory = Callable[[str], BaleBotProtocol]


class BaleTransport(Protocol):
    async def send_message(self, chat_id: str, text: str) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class BaleSettings:
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = DEFAULT_BALE_CHAT_ID
    allowlist: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        normalized_chat = self.chat_id.strip()
        if not normalized_chat.isdigit():
            raise ValueError("Bale chat_id must be numeric")
        normalized_allowlist = frozenset(
            item.strip() for item in self.allowlist if item.strip()
        )
        if any(not item.isdigit() for item in normalized_allowlist):
            raise ValueError("Bale allowlist entries must be numeric chat IDs")
        object.__setattr__(self, "chat_id", normalized_chat)
        object.__setattr__(self, "bot_token", self.bot_token.strip())
        object.__setattr__(self, "allowlist", normalized_allowlist)

    @classmethod
    def from_environment(
        cls, *, default_chat_id: str = DEFAULT_BALE_CHAT_ID
    ) -> "BaleSettings":
        raw_allowlist = os.getenv("ALIx_BALE_CHAT_ALLOWLIST", "")
        allowlist = frozenset(
            value.strip() for value in raw_allowlist.split(",") if value.strip()
        )
        return cls(
            enabled=os.getenv("ALIx_REPORTING_ENABLED", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            bot_token=os.getenv("BALE_BOT_TOKEN", ""),
            chat_id=os.getenv("BALE_CHAT_ID", default_chat_id),
            allowlist=allowlist,
        )


@dataclass(frozen=True, slots=True)
class NotificationResult:
    sent: bool
    reason: str
    chat_id: str


class BaleIOTransport:
    """Adapter to the MIT-licensed BaleIO client; imports it only on send.

    The Bot instance is created per delivery and closed immediately, making
    resource lifetime explicit. Tests inject a fake BotFactory and never use
    sockets or real Bale credentials.
    """

    def __init__(
        self, bot_token: str, *, bot_factory: BotFactory | None = None
    ) -> None:
        token = bot_token.strip()
        if not token:
            raise ValueError("BALE_BOT_TOKEN is required for Bale transport")
        self._token = token
        self._bot_factory = bot_factory or _default_bot_factory

    async def send_message(self, chat_id: str, text: str) -> bool:
        bot = self._bot_factory(self._token)
        try:
            result = await bot.send_message(int(chat_id), text)
            return result is not None
        finally:
            await bot.close()


def _default_bot_factory(token: str) -> BaleBotProtocol:
    module = importlib.import_module("baleio")
    bot_class = getattr(module, "Bot")
    return cast(BaleBotProtocol, bot_class(token))


class BaleNotificationService:
    """Checks enablement and recipient policy before calling the injected transport."""

    def __init__(
        self, settings: BaleSettings, transport: BaleTransport | None = None
    ) -> None:
        self._settings = settings
        self._transport = transport

    @classmethod
    def from_environment(
        cls,
        *,
        default_chat_id: str = DEFAULT_BALE_CHAT_ID,
        transport: BaleTransport | None = None,
    ) -> "BaleNotificationService":
        settings = BaleSettings.from_environment(default_chat_id=default_chat_id)
        resolved_transport = transport
        if resolved_transport is None and settings.bot_token:
            resolved_transport = BaleIOTransport(settings.bot_token)
        return cls(settings, resolved_transport)

    async def send_text(self, text: str) -> NotificationResult:
        chat_id = self._settings.chat_id
        if not self._settings.enabled:
            return NotificationResult(False, "reporting_disabled", chat_id)
        if not self._settings.bot_token:
            return NotificationResult(False, "missing_bot_token", chat_id)
        if chat_id not in self._settings.allowlist:
            return NotificationResult(False, "recipient_not_allowlisted", chat_id)
        if not text.strip() or len(text) > MAX_MESSAGE_LENGTH:
            return NotificationResult(False, "invalid_message", chat_id)
        if self._transport is None:
            return NotificationResult(False, "transport_unavailable", chat_id)
        try:
            sent = await self._transport.send_message(chat_id, text)
        except Exception:
            return NotificationResult(False, "transport_failed", chat_id)
        return NotificationResult(bool(sent), "sent" if sent else "transport_rejected", chat_id)
