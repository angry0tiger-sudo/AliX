"""Bearer-token authentication for the local single-owner deployment."""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass
from typing import Protocol


_TOKEN_MIN_LENGTH = 32
_SUBJECT_PATTERN = re.compile(r"^[A-Za-z0-9._:@/-]{1,160}$")


class AuthenticationError(Exception):
    """Credentials are absent, invalid, or not supported by the configured mode."""


class AuthorizationError(Exception):
    """The authenticated principal lacks permission for the requested resource."""


@dataclass(frozen=True, slots=True)
class Principal:
    subject: str
    platform_admin: bool = False


class Authenticator(Protocol):
    """Authentication boundary injected into the API."""

    def authenticate(self, authorization_header: str | None) -> Principal:
        """Resolve a request header to a verified principal."""
        ...


class SingleUserTokenAuthenticator:
    """Local-only owner token; never falls back to anonymous access."""

    def __init__(self, token: str, subject: str = "local-owner") -> None:
        if len(token) < _TOKEN_MIN_LENGTH or token.strip() != token:
            raise ValueError("ALIx_ADMIN_TOKEN must contain at least 32 characters")
        if not _SUBJECT_PATTERN.fullmatch(subject):
            raise ValueError("invalid_local_owner_subject")
        self._token = token
        self._subject = subject

    def authenticate(self, authorization_header: str | None) -> Principal:
        if authorization_header is None:
            raise AuthenticationError("bearer_token_required")
        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthenticationError("bearer_token_required")
        if not hmac.compare_digest(parts[1], self._token):
            raise AuthenticationError("invalid_bearer_token")
        return Principal(subject=self._subject, platform_admin=True)
