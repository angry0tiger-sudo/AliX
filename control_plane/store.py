"""Transactional tenant, account, worker, command and audit repository."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, insert, or_, select, update
from sqlalchemy.engine import Engine, RowMapping
from sqlalchemy.exc import IntegrityError

from control_plane.auth import AuthorizationError, Principal
from control_plane.db import (
    accounts,
    audit_events,
    initialize_schema,
    memberships,
    runtime_commands,
    tenants,
    worker_slots,
    workers,
)


class ControlPlaneError(Exception):
    """Base class for sanitized service errors."""

    def __init__(self, code: str, status_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class NotFoundError(ControlPlaneError):
    def __init__(self, code: str = "resource_not_found") -> None:
        super().__init__(code, 404)


class ConflictError(ControlPlaneError):
    def __init__(self, code: str) -> None:
        super().__init__(code, 409)


class ForbiddenError(ControlPlaneError):
    def __init__(self, code: str = "forbidden") -> None:
        super().__init__(code, 403)


class InvalidStateError(ControlPlaneError):
    def __init__(self, code: str) -> None:
        super().__init__(code, 422)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _row(row: RowMapping | None) -> dict[str, Any]:
    if row is None:
        raise NotFoundError()
    return dict(row)


def _audit(
    connection: Any,
    principal: Principal,
    action: str,
    resource_type: str,
    resource_id: str,
    tenant_id: str | None,
    details: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        insert(audit_events).values(
            event_id=str(uuid.uuid4()),
            occurred_at=utc_now(),
            actor_subject=principal.subject,
            tenant_id=tenant_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details_json=_json(details or {}),
        )
    )


def _schedule_worker_disables(
    connection: Any,
    principal: Principal,
    tenant_id: str,
    reason: str,
    account_id: str | None = None,
) -> None:
    """Revoke Worker capabilities immediately and queue idempotent container stops."""
    query = select(workers).where(workers.c.tenant_id == tenant_id)
    if account_id is not None:
        query = query.where(workers.c.account_id == account_id)
    rows = connection.execute(query).mappings().all()
    for worker in rows:
        worker_id = str(worker["worker_id"])
        generation = int(worker["generation"])
        now = utc_now()
        if (
            worker["desired_state"] == "DISABLED"
            and worker["observed_state"] == "DISABLED"
            and not worker["container_name"]
        ):
            continue
        connection.execute(
            update(workers)
            .where(workers.c.worker_id == worker_id)
            .values(
                desired_state="DISABLED",
                credential_sha256=None,
                last_heartbeat=None,
                updated_at=now,
            )
        )
        pending_disable = connection.execute(
            select(runtime_commands.c.command_id).where(
                runtime_commands.c.worker_id == worker_id,
                runtime_commands.c.generation == generation,
                runtime_commands.c.action == "DISABLE",
                runtime_commands.c.status.in_(("PENDING", "CLAIMED")),
            )
        ).first()
        command_id: str | None = None
        if pending_disable is None:
            command_id = str(uuid.uuid4())
            connection.execute(
                insert(runtime_commands).values(
                    command_id=command_id,
                    tenant_id=tenant_id,
                    worker_id=worker_id,
                    generation=generation,
                    action="DISABLE",
                    status="PENDING",
                    created_at=now,
                    claimed_by=None,
                    lease_until=None,
                    completed_at=None,
                    error_code=None,
                )
            )
        _audit(
            connection,
            principal,
            f"{reason}.worker_disable_requested",
            "worker",
            worker_id,
            tenant_id,
            {"generation": generation, "command_id": command_id},
        )


def _request_worker_start(
    connection: Any, row: RowMapping, worker_id: str, now: str
) -> int | None:
    desired = str(row["desired_state"])
    observed = str(row["observed_state"])
    if desired == "START_REQUESTED" or observed == "RUNNING":
        return None
    if desired == "DISABLED" or observed == "QUARANTINED":
        raise ConflictError("worker_disabled_or_quarantined")
    generation = int(row["generation"]) + 1
    connection.execute(
        update(workers)
        .where(workers.c.worker_id == worker_id)
        .values(
            desired_state="START_REQUESTED",
            observed_state="STARTING",
            generation=generation,
            credential_sha256=None,
            updated_at=now,
        )
    )
    connection.execute(
        update(worker_slots)
        .where(
            worker_slots.c.tenant_id == row["tenant_id"],
            worker_slots.c.account_id == row["account_id"],
            worker_slots.c.exchange_id == row["exchange_id"],
            worker_slots.c.environment == row["environment"],
        )
        .values(generation=generation, worker_id=worker_id)
    )
    return generation


def _request_worker_stop_or_disable(
    connection: Any,
    row: RowMapping,
    worker_id: str,
    action: str,
    now: str,
) -> int | None:
    desired = str(row["desired_state"])
    observed = str(row["observed_state"])
    if (
        action == "stop"
        and desired == "STOPPED"
        and observed
        in (
            "ABSENT",
            "STOPPED",
            "DISABLED",
        )
    ):
        return None
    if action == "disable" and desired == "DISABLED":
        return None
    generation = int(row["generation"])
    connection.execute(
        update(workers)
        .where(workers.c.worker_id == worker_id)
        .values(
            desired_state="STOP_REQUESTED" if action == "stop" else "DISABLED",
            credential_sha256=None,
            updated_at=now,
        )
    )
    return generation


def _safe_runtime_error_code(error_code: str | None) -> str | None:
    if error_code and re.fullmatch(r"[a-z0-9_]{1,80}", error_code):
        return error_code
    return None


def _apply_runtime_worker_outcome(
    connection: Any,
    command: RowMapping,
    worker: RowMapping,
    success: bool,
    container_name: str | None,
    now: str,
) -> None:
    worker_id = str(command["worker_id"])
    generation = int(command["generation"])
    if generation != int(worker["generation"]):
        return
    if not success:
        connection.execute(
            update(workers)
            .where(
                workers.c.worker_id == worker_id,
                workers.c.generation == generation,
            )
            .values(
                observed_state="QUARANTINED",
                credential_sha256=None,
                updated_at=now,
            )
        )
        return
    action = str(command["action"])
    if action == "START":
        _apply_start_ack(connection, worker, worker_id, generation, container_name, now)
    elif action in ("STOP", "DISABLE"):
        _apply_stop_ack(connection, worker, worker_id, generation, action, now)


def _apply_start_ack(
    connection: Any,
    worker: RowMapping,
    worker_id: str,
    generation: int,
    container_name: str | None,
    now: str,
) -> None:
    if worker["desired_state"] != "START_REQUESTED":
        return
    connection.execute(
        update(workers)
        .where(
            workers.c.worker_id == worker_id,
            workers.c.generation == generation,
            workers.c.desired_state == "START_REQUESTED",
        )
        .values(
            observed_state="RUNNING" if worker["last_heartbeat"] else "STARTING",
            container_name=container_name,
            updated_at=now,
        )
    )


def _apply_stop_ack(
    connection: Any,
    worker: RowMapping,
    worker_id: str,
    generation: int,
    action: str,
    now: str,
) -> None:
    expected_desired = "STOP_REQUESTED" if action == "STOP" else "DISABLED"
    if worker["desired_state"] != expected_desired:
        return
    connection.execute(
        update(workers)
        .where(
            workers.c.worker_id == worker_id,
            workers.c.generation == generation,
            workers.c.desired_state == expected_desired,
        )
        .values(
            desired_state="DISABLED" if action == "DISABLE" else "STOPPED",
            observed_state="DISABLED" if action == "DISABLE" else "STOPPED",
            container_name=None,
            credential_sha256=None,
            last_heartbeat=None,
            updated_at=now,
        )
    )


def _require_active_worker_account(
    connection: Any,
    tenant_id: str,
    account_id: str,
    exchange_id: str,
    environment: str,
) -> None:
    account = (
        connection.execute(
            select(accounts).where(
                accounts.c.tenant_id == tenant_id,
                accounts.c.account_id == account_id,
                accounts.c.exchange_id == exchange_id,
                accounts.c.environment == environment,
            )
        )
        .mappings()
        .first()
    )
    if account is None:
        raise NotFoundError("account_exchange_binding_not_found")
    if account["status"] != "ACTIVE":
        raise InvalidStateError("account_not_active")


def _reusable_worker_slot_generation(connection: Any, slot: RowMapping) -> int:
    previous = (
        connection.execute(
            select(
                workers.c.desired_state,
                workers.c.observed_state,
                workers.c.generation,
                workers.c.container_name,
            ).where(workers.c.worker_id == slot["worker_id"])
        )
        .mappings()
        .first()
    )
    reusable_disabled_slot = (
        previous is not None
        and previous["desired_state"] == "DISABLED"
        and previous["observed_state"] == "DISABLED"
        and not previous["container_name"]
    )
    reusable_stopped_slot = (
        previous is not None
        and int(previous["generation"]) > 0
        and previous["desired_state"] == "STOPPED"
        and previous["observed_state"] in ("ABSENT", "STOPPED", "FAILED")
        and not previous["container_name"]
    )
    if not (reusable_disabled_slot or reusable_stopped_slot):
        raise ConflictError("worker_slot_already_occupied")
    return int(slot["generation"]) + 1


class ControlPlaneStore:
    """Authorization-aware persistence; callers never provide effective scope alone."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        initialize_schema(engine)

    def ping(self) -> bool:
        with self.engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
        return True

    def _require_tenant(
        self,
        connection: Any,
        principal: Principal,
        tenant_id: str,
        allowed_roles: tuple[str, ...],
        require_active: bool = True,
    ) -> None:
        tenant = connection.execute(
            select(tenants.c.status).where(tenants.c.tenant_id == tenant_id)
        ).first()
        if tenant is None:
            raise NotFoundError("tenant_not_found")
        tenant_status = str(tenant._mapping["status"])
        if tenant_status != "ACTIVE" and not principal.platform_admin:
            raise ForbiddenError("tenant_not_active")
        if require_active and tenant_status != "ACTIVE":
            raise ForbiddenError("tenant_not_active")
        if principal.platform_admin:
            return
        membership = connection.execute(
            select(memberships.c.role, memberships.c.status).where(
                memberships.c.tenant_id == tenant_id,
                memberships.c.subject == principal.subject,
            )
        ).first()
        if (
            membership is None
            or membership._mapping["status"] != "ACTIVE"
            or membership._mapping["role"] not in allowed_roles
        ):
            raise ForbiddenError()

    def create_tenant(self, principal: Principal, name: str) -> dict[str, Any]:
        if not principal.platform_admin:
            raise ForbiddenError("platform_admin_required")
        tenant_id = str(uuid.uuid4())
        now = utc_now()
        with self.engine.begin() as connection:
            connection.execute(
                insert(tenants).values(
                    tenant_id=tenant_id, name=name, status="ACTIVE", created_at=now
                )
            )
            connection.execute(
                insert(memberships).values(
                    tenant_id=tenant_id,
                    subject=principal.subject,
                    role="tenant_owner",
                    status="ACTIVE",
                    created_at=now,
                )
            )
            _audit(
                connection,
                principal,
                "tenant.created",
                "tenant",
                tenant_id,
                tenant_id,
                {"name": name},
            )
        return self.get_tenant(principal, tenant_id)

    def disable_tenant(self, principal: Principal, tenant_id: str) -> dict[str, Any]:
        if not principal.platform_admin:
            raise ForbiddenError("platform_admin_required")
        with self.engine.begin() as connection:
            row = connection.execute(
                select(tenants.c.status).where(tenants.c.tenant_id == tenant_id)
            ).first()
            if row is None:
                raise NotFoundError("tenant_not_found")
            if row._mapping["status"] != "DISABLED":
                connection.execute(
                    update(tenants)
                    .where(tenants.c.tenant_id == tenant_id)
                    .values(status="DISABLED")
                )
                _audit(
                    connection,
                    principal,
                    "tenant.disabled",
                    "tenant",
                    tenant_id,
                    tenant_id,
                )
            _schedule_worker_disables(connection, principal, tenant_id, "tenant")
        return self.get_tenant(principal, tenant_id)

    def get_tenant(self, principal: Principal, tenant_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            self._require_tenant(
                connection,
                principal,
                tenant_id,
                ("tenant_owner", "tenant_operator", "tenant_reader"),
                require_active=False,
            )
            row = (
                connection.execute(
                    select(tenants).where(tenants.c.tenant_id == tenant_id)
                )
                .mappings()
                .first()
            )
            return _row(row)

    def list_tenants(self, principal: Principal) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            query = select(tenants)
            if not principal.platform_admin:
                query = query.join(
                    memberships, memberships.c.tenant_id == tenants.c.tenant_id
                ).where(
                    memberships.c.subject == principal.subject,
                    memberships.c.status == "ACTIVE",
                    tenants.c.status == "ACTIVE",
                )
            return [dict(item) for item in connection.execute(query).mappings().all()]

    def add_membership(
        self, principal: Principal, tenant_id: str, subject: str, role: str
    ) -> dict[str, Any]:
        with self.engine.begin() as connection:
            self._require_tenant(connection, principal, tenant_id, ("tenant_owner",))
            existing = connection.execute(
                select(memberships.c.status).where(
                    memberships.c.tenant_id == tenant_id,
                    memberships.c.subject == subject,
                )
            ).first()
            if existing is not None:
                raise ConflictError("membership_already_exists")
            now = utc_now()
            connection.execute(
                insert(memberships).values(
                    tenant_id=tenant_id,
                    subject=subject,
                    role=role,
                    status="ACTIVE",
                    created_at=now,
                )
            )
            _audit(
                connection,
                principal,
                "membership.created",
                "membership",
                f"{tenant_id}:{subject}",
                tenant_id,
                {"role": role},
            )
        return {
            "tenant_id": tenant_id,
            "subject": subject,
            "role": role,
            "status": "ACTIVE",
        }

    def revoke_membership(
        self, principal: Principal, tenant_id: str, subject: str
    ) -> None:
        with self.engine.begin() as connection:
            self._require_tenant(connection, principal, tenant_id, ("tenant_owner",))
            result = connection.execute(
                update(memberships)
                .where(
                    memberships.c.tenant_id == tenant_id,
                    memberships.c.subject == subject,
                    memberships.c.status == "ACTIVE",
                )
                .values(status="REVOKED")
            )
            if result.rowcount != 1:
                raise NotFoundError("active_membership_not_found")
            _audit(
                connection,
                principal,
                "membership.revoked",
                "membership",
                f"{tenant_id}:{subject}",
                tenant_id,
            )

    def create_account(
        self,
        principal: Principal,
        tenant_id: str,
        exchange_id: str,
        environment: str,
        account_alias: str,
        secret_ref_id: str | None,
    ) -> dict[str, Any]:
        account_id = str(uuid.uuid4())
        now = utc_now()
        with self.engine.begin() as connection:
            self._require_tenant(connection, principal, tenant_id, ("tenant_owner",))
            connection.execute(
                insert(accounts).values(
                    account_id=account_id,
                    tenant_id=tenant_id,
                    exchange_id=exchange_id,
                    environment=environment,
                    account_alias=account_alias,
                    status="REGISTERED_DISABLED",
                    secret_ref_id=secret_ref_id,
                    created_at=now,
                )
            )
            _audit(
                connection,
                principal,
                "account.registered",
                "account",
                account_id,
                tenant_id,
                {"exchange_id": exchange_id, "environment": environment},
            )
        return self.get_account(principal, tenant_id, account_id)

    def enable_account(
        self, principal: Principal, tenant_id: str, account_id: str
    ) -> dict[str, Any]:
        with self.engine.begin() as connection:
            self._require_tenant(connection, principal, tenant_id, ("tenant_owner",))
            account = connection.execute(
                select(accounts.c.status).where(
                    accounts.c.tenant_id == tenant_id,
                    accounts.c.account_id == account_id,
                )
            ).first()
            if account is None:
                raise NotFoundError("account_not_found")
            if account._mapping["status"] == "DISABLED":
                raise InvalidStateError("disabled_account_requires_new_binding")
            if account._mapping["status"] != "ACTIVE":
                connection.execute(
                    update(accounts)
                    .where(
                        accounts.c.tenant_id == tenant_id,
                        accounts.c.account_id == account_id,
                    )
                    .values(status="ACTIVE")
                )
                _audit(
                    connection,
                    principal,
                    "account.enabled",
                    "account",
                    account_id,
                    tenant_id,
                )
        return self.get_account(principal, tenant_id, account_id)

    def disable_account(
        self, principal: Principal, tenant_id: str, account_id: str
    ) -> dict[str, Any]:
        with self.engine.begin() as connection:
            self._require_tenant(connection, principal, tenant_id, ("tenant_owner",))
            account = connection.execute(
                select(accounts.c.status).where(
                    accounts.c.tenant_id == tenant_id,
                    accounts.c.account_id == account_id,
                )
            ).first()
            if account is None:
                raise NotFoundError("account_not_found")
            if account._mapping["status"] != "DISABLED":
                connection.execute(
                    update(accounts)
                    .where(
                        accounts.c.tenant_id == tenant_id,
                        accounts.c.account_id == account_id,
                    )
                    .values(status="DISABLED")
                )
                _audit(
                    connection,
                    principal,
                    "account.disabled",
                    "account",
                    account_id,
                    tenant_id,
                )
            _schedule_worker_disables(
                connection, principal, tenant_id, "account", account_id=account_id
            )
        return self.get_account(principal, tenant_id, account_id)

    def get_account(
        self, principal: Principal, tenant_id: str, account_id: str
    ) -> dict[str, Any]:
        with self.engine.connect() as connection:
            self._require_tenant(
                connection,
                principal,
                tenant_id,
                ("tenant_owner", "tenant_operator", "tenant_reader"),
                require_active=False,
            )
            row = (
                connection.execute(
                    select(accounts).where(
                        accounts.c.tenant_id == tenant_id,
                        accounts.c.account_id == account_id,
                    )
                )
                .mappings()
                .first()
            )
            return _row(row)

    def list_accounts(
        self, principal: Principal, tenant_id: str
    ) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            self._require_tenant(
                connection,
                principal,
                tenant_id,
                ("tenant_owner", "tenant_operator", "tenant_reader"),
                require_active=False,
            )
            rows = (
                connection.execute(
                    select(accounts)
                    .where(accounts.c.tenant_id == tenant_id)
                    .order_by(accounts.c.created_at)
                )
                .mappings()
                .all()
            )
            return [dict(item) for item in rows]

    def create_worker(
        self,
        principal: Principal,
        tenant_id: str,
        account_id: str,
        exchange_id: str,
        environment: str,
        strategy_revision: str,
        profile_id: str,
    ) -> dict[str, Any]:
        if environment not in ("local", "test"):
            raise InvalidStateError("only_nontrading_environments_are_allowed")
        worker_id = str(uuid.uuid4())
        now = utc_now()
        try:
            with self.engine.begin() as connection:
                self._require_tenant(
                    connection,
                    principal,
                    tenant_id,
                    ("tenant_owner", "tenant_operator"),
                )
                _require_active_worker_account(
                    connection, tenant_id, account_id, exchange_id, environment
                )
                slot_filter = and_(
                    worker_slots.c.tenant_id == tenant_id,
                    worker_slots.c.account_id == account_id,
                    worker_slots.c.exchange_id == exchange_id,
                    worker_slots.c.environment == environment,
                )
                slot = (
                    connection.execute(select(worker_slots).where(slot_filter))
                    .mappings()
                    .first()
                )
                generation = 0
                if slot is not None:
                    generation = _reusable_worker_slot_generation(connection, slot)
                connection.execute(
                    insert(workers).values(
                        worker_id=worker_id,
                        tenant_id=tenant_id,
                        account_id=account_id,
                        exchange_id=exchange_id,
                        environment=environment,
                        strategy_revision=strategy_revision,
                        profile_id=profile_id,
                        desired_state="STOPPED",
                        observed_state="ABSENT",
                        generation=generation,
                        container_name=None,
                        credential_sha256=None,
                        last_heartbeat=None,
                        created_at=now,
                        updated_at=now,
                    )
                )
                slot_values = {
                    "tenant_id": tenant_id,
                    "account_id": account_id,
                    "exchange_id": exchange_id,
                    "environment": environment,
                    "generation": generation,
                    "worker_id": worker_id,
                }
                if slot is None:
                    connection.execute(insert(worker_slots).values(**slot_values))
                else:
                    connection.execute(
                        update(worker_slots).where(slot_filter).values(**slot_values)
                    )
                _audit(
                    connection,
                    principal,
                    "worker.registered",
                    "worker",
                    worker_id,
                    tenant_id,
                    {
                        "exchange_id": exchange_id,
                        "environment": environment,
                        "profile_id": profile_id,
                    },
                )
        except IntegrityError as exc:
            raise ConflictError("worker_slot_or_binding_conflict") from exc
        return self.get_worker(principal, tenant_id, worker_id)

    def get_worker(
        self, principal: Principal, tenant_id: str, worker_id: str
    ) -> dict[str, Any]:
        with self.engine.connect() as connection:
            self._require_tenant(
                connection,
                principal,
                tenant_id,
                ("tenant_owner", "tenant_operator", "tenant_reader"),
                require_active=False,
            )
            row = (
                connection.execute(
                    select(workers).where(
                        workers.c.tenant_id == tenant_id,
                        workers.c.worker_id == worker_id,
                    )
                )
                .mappings()
                .first()
            )
            return _row(row)

    def list_workers(
        self, principal: Principal, tenant_id: str
    ) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            self._require_tenant(
                connection,
                principal,
                tenant_id,
                ("tenant_owner", "tenant_operator", "tenant_reader"),
                require_active=False,
            )
            rows = (
                connection.execute(
                    select(workers)
                    .where(workers.c.tenant_id == tenant_id)
                    .order_by(workers.c.created_at)
                )
                .mappings()
                .all()
            )
            return [dict(item) for item in rows]

    def transition_worker(
        self,
        principal: Principal,
        tenant_id: str,
        worker_id: str,
        action: str,
    ) -> dict[str, Any]:
        command_action = {"start": "START", "stop": "STOP", "disable": "DISABLE"}[
            action
        ]
        now = utc_now()
        with self.engine.begin() as connection:
            self._require_tenant(
                connection, principal, tenant_id, ("tenant_owner", "tenant_operator")
            )
            row = (
                connection.execute(
                    select(workers).where(
                        workers.c.tenant_id == tenant_id,
                        workers.c.worker_id == worker_id,
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                raise NotFoundError("worker_not_found")
            if action == "start":
                generation = _request_worker_start(connection, row, worker_id, now)
            else:
                generation = _request_worker_stop_or_disable(
                    connection, row, worker_id, action, now
                )
            if generation is None:
                return dict(row)
            command_id = str(uuid.uuid4())
            connection.execute(
                insert(runtime_commands).values(
                    command_id=command_id,
                    tenant_id=tenant_id,
                    worker_id=worker_id,
                    generation=generation,
                    action=command_action,
                    status="PENDING",
                    created_at=now,
                    claimed_by=None,
                    lease_until=None,
                    completed_at=None,
                    error_code=None,
                )
            )
            _audit(
                connection,
                principal,
                f"worker.{action}_requested",
                "worker",
                worker_id,
                tenant_id,
                {"generation": generation, "command_id": command_id},
            )
        return self.get_worker(principal, tenant_id, worker_id)

    def claim_runtime_commands(
        self,
        dispatcher_id: str,
        limit: int,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        current = now or datetime.now(timezone.utc)
        current_s = current.isoformat(timespec="seconds")
        lease_s = (current + timedelta(seconds=lease_seconds)).isoformat(
            timespec="seconds"
        )
        claimed: list[dict[str, Any]] = []
        eligible = or_(
            runtime_commands.c.status == "PENDING",
            and_(
                runtime_commands.c.status == "CLAIMED",
                runtime_commands.c.lease_until <= current_s,
            ),
        )
        with self.engine.begin() as connection:
            candidates = connection.execute(
                select(runtime_commands.c.command_id)
                .where(eligible)
                .order_by(runtime_commands.c.created_at)
                .limit(limit)
            ).all()
            for candidate in candidates:
                command_id = str(candidate._mapping["command_id"])
                updated = connection.execute(
                    update(runtime_commands)
                    .where(runtime_commands.c.command_id == command_id, eligible)
                    .values(
                        status="CLAIMED",
                        claimed_by=dispatcher_id,
                        lease_until=lease_s,
                    )
                )
                if updated.rowcount != 1:
                    continue
                row = (
                    connection.execute(
                        select(runtime_commands).where(
                            runtime_commands.c.command_id == command_id
                        )
                    )
                    .mappings()
                    .first()
                )
                if row is not None:
                    claimed.append(dict(row))
        return claimed

    def register_worker_credential(
        self, worker_id: str, generation: int, credential_sha256: str
    ) -> None:
        now = utc_now()
        with self.engine.begin() as connection:
            result = connection.execute(
                update(workers)
                .where(
                    workers.c.worker_id == worker_id,
                    workers.c.generation == generation,
                    workers.c.desired_state == "START_REQUESTED",
                    workers.c.observed_state.in_(("STARTING", "RUNNING")),
                )
                .values(credential_sha256=credential_sha256, updated_at=now)
            )
            if result.rowcount != 1:
                raise ConflictError("worker_generation_or_state_mismatch")

    def heartbeat_worker(
        self, worker_id: str, generation: int, bearer_token: str | None
    ) -> None:
        if bearer_token is None:
            raise AuthorizationError("worker_token_required")
        token_hash = hashlib.sha256(bearer_token.encode("utf-8")).hexdigest()
        now = utc_now()
        with self.engine.begin() as connection:
            row = (
                connection.execute(
                    select(
                        workers.c.tenant_id,
                        workers.c.generation,
                        workers.c.credential_sha256,
                        workers.c.desired_state,
                        tenants.c.status.label("tenant_status"),
                    )
                    .join(tenants, tenants.c.tenant_id == workers.c.tenant_id)
                    .where(workers.c.worker_id == worker_id)
                )
                .mappings()
                .first()
            )
            if row is None:
                raise NotFoundError("worker_not_found")
            expected_hash = row["credential_sha256"]
            if (
                not expected_hash
                or not hmac.compare_digest(str(expected_hash), token_hash)
                or int(row["generation"]) != generation
                or row["desired_state"] != "START_REQUESTED"
                or row["tenant_status"] != "ACTIVE"
            ):
                raise AuthorizationError("worker_identity_or_scope_invalid")
            result = connection.execute(
                update(workers)
                .where(
                    workers.c.worker_id == worker_id,
                    workers.c.generation == generation,
                    workers.c.credential_sha256 == expected_hash,
                    workers.c.desired_state == "START_REQUESTED",
                )
                .values(
                    observed_state="RUNNING",
                    last_heartbeat=now,
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                raise AuthorizationError("worker_identity_or_scope_invalid")

    def claimed_command_owner(self, command_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(runtime_commands).where(
                        runtime_commands.c.command_id == command_id,
                        runtime_commands.c.status == "CLAIMED",
                    )
                )
                .mappings()
                .first()
            )
            return None if row is None else dict(row)

    def acknowledge_runtime_command(
        self,
        command_id: str,
        dispatcher_id: str,
        success: bool,
        container_name: str | None,
        error_code: str | None,
    ) -> None:
        now = utc_now()
        with self.engine.begin() as connection:
            command = (
                connection.execute(
                    select(runtime_commands).where(
                        runtime_commands.c.command_id == command_id,
                        runtime_commands.c.status == "CLAIMED",
                        runtime_commands.c.claimed_by == dispatcher_id,
                    )
                )
                .mappings()
                .first()
            )
            if command is None:
                raise ConflictError("runtime_command_lease_not_owned")
            worker = (
                connection.execute(
                    select(workers).where(workers.c.worker_id == command["worker_id"])
                )
                .mappings()
                .first()
            )
            if worker is None:
                raise NotFoundError("worker_not_found")
            safe_error = _safe_runtime_error_code(error_code)
            connection.execute(
                update(runtime_commands)
                .where(runtime_commands.c.command_id == command_id)
                .values(
                    status="SUCCEEDED" if success else "FAILED",
                    completed_at=now,
                    error_code=None if success else (safe_error or "runtime_failed"),
                    lease_until=None,
                )
            )
            _apply_runtime_worker_outcome(
                connection,
                command,
                worker,
                success,
                container_name,
                now,
            )
            _audit(
                connection,
                Principal(subject=f"runtime:{dispatcher_id}", platform_admin=True),
                "runtime_command.acknowledged",
                "runtime_command",
                command_id,
                str(command["tenant_id"]),
                {
                    "action": str(command["action"]),
                    "generation": int(command["generation"]),
                    "success": success,
                    "error_code": None if success else (safe_error or "runtime_failed"),
                },
            )

    def list_audit_events(
        self, principal: Principal, tenant_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            self._require_tenant(connection, principal, tenant_id, ("tenant_owner",))
            rows = (
                connection.execute(
                    select(audit_events)
                    .where(audit_events.c.tenant_id == tenant_id)
                    .order_by(audit_events.c.occurred_at.desc())
                    .limit(limit)
                )
                .mappings()
                .all()
            )
            result: list[dict[str, Any]] = []
            for item in rows:
                record = dict(item)
                record["details"] = json.loads(str(record.pop("details_json")))
                result.append(record)
            return result
