"""Versioned SQLAlchemy schema for PostgreSQL control state and SQLite tests."""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine


SCHEMA_VERSION = "1"
metadata = MetaData()

tenants = Table(
    "tenants",
    metadata,
    Column("tenant_id", String(80), primary_key=True),
    Column("name", String(128), nullable=False),
    Column("status", String(20), nullable=False),
    Column("created_at", String(40), nullable=False),
    CheckConstraint("status IN ('ACTIVE', 'DISABLED')", name="ck_tenants_status"),
)

memberships = Table(
    "tenant_memberships",
    metadata,
    Column("tenant_id", String(80), ForeignKey("tenants.tenant_id"), primary_key=True),
    Column("subject", String(160), primary_key=True),
    Column("role", String(30), nullable=False),
    Column("status", String(20), nullable=False),
    Column("created_at", String(40), nullable=False),
    CheckConstraint(
        "role IN ('tenant_owner', 'tenant_operator', 'tenant_reader')",
        name="ck_membership_role",
    ),
    CheckConstraint("status IN ('ACTIVE', 'REVOKED')", name="ck_membership_status"),
)

accounts = Table(
    "accounts",
    metadata,
    Column("account_id", String(80), primary_key=True),
    Column("tenant_id", String(80), ForeignKey("tenants.tenant_id"), nullable=False),
    Column("exchange_id", String(40), nullable=False),
    Column("environment", String(20), nullable=False),
    Column("account_alias", String(80), nullable=False),
    Column("status", String(30), nullable=False),
    Column("secret_ref_id", String(200), nullable=True),
    Column("created_at", String(40), nullable=False),
    UniqueConstraint(
        "tenant_id",
        "account_id",
        "exchange_id",
        "environment",
        name="uq_account_tenant_binding",
    ),
    CheckConstraint(
        "environment IN ('local', 'test')",
        name="ck_account_nontrading_environment",
    ),
    CheckConstraint(
        "status IN ('REGISTERED_DISABLED', 'DISABLED')",
        name="ck_account_status",
    ),
)

workers = Table(
    "workers",
    metadata,
    Column("worker_id", String(80), primary_key=True),
    Column("tenant_id", String(80), ForeignKey("tenants.tenant_id"), nullable=False),
    Column("account_id", String(80), nullable=False),
    Column("exchange_id", String(40), nullable=False),
    Column("environment", String(20), nullable=False),
    Column("strategy_revision", String(128), nullable=False),
    Column("profile_id", String(80), nullable=False),
    Column("desired_state", String(30), nullable=False),
    Column("observed_state", String(30), nullable=False),
    Column("generation", Integer, nullable=False),
    Column("container_name", String(120), nullable=True),
    Column("credential_sha256", String(64), nullable=True),
    Column("last_heartbeat", String(40), nullable=True),
    Column("created_at", String(40), nullable=False),
    Column("updated_at", String(40), nullable=False),
    ForeignKeyConstraint(
        ["tenant_id", "account_id", "exchange_id", "environment"],
        [
            "accounts.tenant_id",
            "accounts.account_id",
            "accounts.exchange_id",
            "accounts.environment",
        ],
        name="fk_worker_account_binding",
    ),
    CheckConstraint(
        "environment IN ('local', 'test')",
        name="ck_worker_nontrading_environment",
    ),
    CheckConstraint(
        "desired_state IN ('STOPPED', 'START_REQUESTED', 'STOP_REQUESTED', 'DISABLED')",
        name="ck_worker_desired_state",
    ),
    CheckConstraint(
        "observed_state IN "
        "('ABSENT', 'STARTING', 'RUNNING', 'STOPPED', 'DISABLED', 'FAILED', 'QUARANTINED')",
        name="ck_worker_observed_state",
    ),
    CheckConstraint("generation >= 0", name="ck_worker_generation"),
)

audit_events = Table(
    "audit_events",
    metadata,
    Column("event_id", String(80), primary_key=True),
    Column("occurred_at", String(40), nullable=False),
    Column("actor_subject", String(160), nullable=False),
    Column("tenant_id", String(80), ForeignKey("tenants.tenant_id"), nullable=True),
    Column("action", String(100), nullable=False),
    Column("resource_type", String(50), nullable=False),
    Column("resource_id", String(160), nullable=False),
    Column("details_json", String(4000), nullable=False),
)

worker_slots = Table(
    "worker_slots",
    metadata,
    Column("tenant_id", String(80), primary_key=True),
    Column("account_id", String(80), primary_key=True),
    Column("exchange_id", String(40), primary_key=True),
    Column("environment", String(20), primary_key=True),
    Column("generation", Integer, nullable=False),
    Column("worker_id", String(80), nullable=False),
    ForeignKeyConstraint(
        ["tenant_id", "account_id", "exchange_id", "environment"],
        [
            "accounts.tenant_id",
            "accounts.account_id",
            "accounts.exchange_id",
            "accounts.environment",
        ],
        name="fk_worker_slot_account",
    ),
)

runtime_commands = Table(
    "runtime_commands",
    metadata,
    Column("command_id", String(80), primary_key=True),
    Column("tenant_id", String(80), ForeignKey("tenants.tenant_id"), nullable=False),
    Column("worker_id", String(80), ForeignKey("workers.worker_id"), nullable=False),
    Column("generation", Integer, nullable=False),
    Column("action", String(20), nullable=False),
    Column("status", String(20), nullable=False),
    Column("created_at", String(40), nullable=False),
    Column("claimed_by", String(100), nullable=True),
    Column("lease_until", String(40), nullable=True),
    Column("completed_at", String(40), nullable=True),
    Column("error_code", String(80), nullable=True),
    CheckConstraint("action IN ('START', 'STOP', 'DISABLE')", name="ck_runtime_action"),
    CheckConstraint(
        "status IN ('PENDING', 'CLAIMED', 'SUCCEEDED', 'FAILED')",
        name="ck_runtime_status",
    ),
)

Index("ix_memberships_subject_status", memberships.c.subject, memberships.c.status)
Index("ix_accounts_tenant", accounts.c.tenant_id)
Index("ix_workers_tenant", workers.c.tenant_id)
Index(
    "ix_runtime_commands_status_created",
    runtime_commands.c.status,
    runtime_commands.c.created_at,
)

schema_meta = Table(
    "schema_meta",
    metadata,
    Column("schema_key", String(80), primary_key=True),
    Column("schema_value", String(80), nullable=False),
)


def create_engine_for_url(database_url: str) -> Engine:
    """Create a database engine and enforce SQLite foreign-key checks in tests."""
    engine = create_engine(database_url, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(
            dbapi_connection: Any, connection_record: Any
        ) -> None:
            del connection_record
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def initialize_schema(engine: Engine) -> None:
    """Create the initial schema and refuse to open a future unknown schema."""
    metadata.create_all(engine)
    with engine.begin() as connection:
        row = connection.execute(
            schema_meta.select().where(schema_meta.c.schema_key == "control_plane")
        ).first()
        if row is None:
            connection.execute(
                schema_meta.insert().values(
                    schema_key="control_plane", schema_value=SCHEMA_VERSION
                )
            )
        elif row._mapping["schema_value"] != SCHEMA_VERSION:
            raise RuntimeError("unsupported_control_plane_schema_version")
