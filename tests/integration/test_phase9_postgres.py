from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from control_plane.auth import Principal
from control_plane.db import audit_events, create_engine_for_url, schema_meta
from control_plane.store import ConflictError, ControlPlaneStore

POSTGRES_URL = os.getenv("ALIX_TEST_POSTGRES_URL", "")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="ALIX_TEST_POSTGRES_URL is set only by the dedicated PostgreSQL integration job",
)


def test_real_postgresql_control_plane_lifecycle_and_constraints() -> None:
    engine = create_engine_for_url(POSTGRES_URL)
    try:
        assert engine.dialect.name == "postgresql"
        store = ControlPlaneStore(engine)
        assert store.ping()

        owner = Principal("phase9-postgres-fixture-owner", platform_admin=True)
        tenant = store.create_tenant(owner, "Phase 9 disposable PostgreSQL fixture")
        tenant_id = str(tenant["tenant_id"])
        account = store.create_account(
            owner,
            tenant_id,
            "synthetic-exchange",
            "test",
            "postgres-fixture-account",
            "secret-ref:synthetic-only",
        )
        account_id = str(account["account_id"])
        assert account["status"] == "REGISTERED_DISABLED"
        enabled = store.enable_account(owner, tenant_id, account_id)
        assert enabled["status"] == "ACTIVE"

        worker = store.create_worker(
            owner,
            tenant_id,
            account_id,
            "synthetic-exchange",
            "test",
            "synthetic-strategy-v1",
            "heartbeat-only",
        )
        assert worker["tenant_id"] == tenant_id
        assert worker["account_id"] == account_id
        assert worker["desired_state"] == "STOPPED"
        assert worker["observed_state"] == "ABSENT"
        assert worker["container_name"] is None

        with pytest.raises(ConflictError, match="worker_slot_already_occupied"):
            store.create_worker(
                owner,
                tenant_id,
                account_id,
                "synthetic-exchange",
                "test",
                "synthetic-strategy-v2",
                "heartbeat-only",
            )
        workers = store.list_workers(owner, tenant_id)
        assert [item["worker_id"] for item in workers] == [worker["worker_id"]]

        with engine.connect() as connection:
            schema_row = (
                connection.execute(
                    schema_meta.select().where(
                        schema_meta.c.schema_key == "control_plane"
                    )
                )
                .mappings()
                .one()
            )
            assert schema_row["schema_value"] == "1"
            audit_count = connection.execute(
                audit_events.select().where(audit_events.c.tenant_id == tenant_id)
            ).fetchall()
            assert len(audit_count) >= 3
    finally:
        engine.dispose()


def test_concurrent_worker_creation_for_one_slot_has_exactly_one_winner() -> None:
    engine = create_engine_for_url(POSTGRES_URL)
    try:
        store = ControlPlaneStore(engine)
        owner = Principal("phase9-concurrent-postgres-owner", platform_admin=True)
        tenant = store.create_tenant(owner, "Phase 9 concurrency fixture")
        tenant_id = str(tenant["tenant_id"])
        account = store.create_account(
            owner,
            tenant_id,
            "synthetic-exchange",
            "test",
            "postgres-concurrency-account",
            "secret-ref:synthetic-only",
        )
        account_id = str(account["account_id"])
        store.enable_account(owner, tenant_id, account_id)
        barrier = Barrier(2)

        def create_worker(revision: str) -> dict[str, object] | None:
            barrier.wait(timeout=10)
            try:
                return store.create_worker(
                    owner,
                    tenant_id,
                    account_id,
                    "synthetic-exchange",
                    "test",
                    revision,
                    "heartbeat-only",
                )
            except ConflictError:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(create_worker, ("synthetic-v1", "synthetic-v2")))

        assert sum(item is not None for item in outcomes) == 1
        assert sum(item is None for item in outcomes) == 1
        assert len(store.list_workers(owner, tenant_id)) == 1
    finally:
        engine.dispose()
