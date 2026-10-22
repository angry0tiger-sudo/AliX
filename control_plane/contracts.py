"""Strict API contracts. Secret values are not accepted by these models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Role = Literal["tenant_owner", "tenant_operator", "tenant_reader"]
LocalEnvironment = Literal["local", "test"]
LifecycleAction = Literal["start", "stop", "disable"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TenantCreate(StrictModel):
    name: str = Field(min_length=1, max_length=128)


class MembershipCreate(StrictModel):
    subject: str = Field(min_length=1, max_length=160)
    role: Role


class AccountCreate(StrictModel):
    exchange_id: str = Field(pattern=r"^[a-z0-9_-]{1,40}$")
    environment: LocalEnvironment
    account_alias: str = Field(min_length=1, max_length=80)
    secret_ref_id: str | None = Field(
        default=None, pattern=r"^secret://[A-Za-z0-9._/-]{1,180}$"
    )


class WorkerCreate(StrictModel):
    account_id: str = Field(min_length=1, max_length=80)
    exchange_id: str = Field(pattern=r"^[a-z0-9_-]{1,40}$")
    environment: LocalEnvironment
    strategy_revision: str = Field(min_length=1, max_length=128)
    profile_id: str = Field(default="freqtrade-isolated", min_length=1, max_length=80)


class LifecycleRequest(StrictModel):
    action: LifecycleAction


class RuntimeClaimRequest(StrictModel):
    dispatcher_id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,100}$")
    lease_seconds: int = Field(default=30, ge=10, le=120)
    limit: int = Field(default=10, ge=1, le=50)


class RuntimeCredentialRequest(StrictModel):
    generation: int = Field(ge=1)
    credential_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeAckRequest(StrictModel):
    success: bool
    container_name: str | None = Field(default=None, max_length=120)
    error_code: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,80}$")


class WorkerHeartbeatRequest(StrictModel):
    generation: int = Field(ge=1)
