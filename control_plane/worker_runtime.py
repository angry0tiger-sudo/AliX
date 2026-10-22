"""Non-trading worker supervisor stub with private SQLite state and scoped heartbeat."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import quote


class WorkerAuthorizationFailure(RuntimeError):
    """The worker identity is revoked, stale, or scoped incorrectly."""


class WorkerRuntimeFailure(RuntimeError):
    """The local worker failed closed."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_worker_id(value: str) -> str:
    if (
        not value
        or len(value) > 80
        or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for ch in value
        )
    ):
        raise ValueError("invalid_worker_id")
    return value


def _send_heartbeat(
    base_url: str, worker_id: str, token: str, generation: int, timeout: float = 3.0
) -> None:
    url = f"{base_url.rstrip('/')}/v1/internal/workers/{quote(worker_id, safe='')}/heartbeat"
    payload = json.dumps({"generation": generation}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 204:
                raise WorkerRuntimeFailure("heartbeat_response_unexpected")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404, 409):
            raise WorkerAuthorizationFailure("worker_identity_rejected") from None
        raise OSError("heartbeat_request_failed") from None
    except urllib.error.URLError:
        raise OSError("heartbeat_request_failed") from None


def _initialize_worker_state(path: Path, worker_id: str, generation: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS worker_identity "
            "(singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
            "worker_id TEXT NOT NULL, generation INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS worker_heartbeat "
            "(sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
            "heartbeat_at TEXT NOT NULL, generation INTEGER NOT NULL)"
        )
        row = connection.execute(
            "SELECT worker_id, generation FROM worker_identity WHERE singleton=1"
        ).fetchone()
        if row is not None and (row[0] != worker_id or int(row[1]) != generation):
            raise WorkerRuntimeFailure("worker_state_identity_mismatch")
        if row is None:
            connection.execute(
                "INSERT INTO worker_identity(singleton, worker_id, generation) VALUES(1, ?, ?)",
                (worker_id, generation),
            )
        connection.commit()
    finally:
        connection.close()


def _record_heartbeat(path: Path, generation: int) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "INSERT INTO worker_heartbeat(heartbeat_at, generation) VALUES(?, ?)",
            (_timestamp(), generation),
        )
        connection.commit()
    finally:
        connection.close()


def run_worker(
    worker_id: str,
    token: str,
    generation: int,
    base_url: str,
    database_path: str,
    heartbeat_sender: Callable[[str, str, str, int], None] = _send_heartbeat,
    interval_seconds: float = 5.0,
    iterations: int | None = None,
) -> int:
    """Run only an inert identity/heartbeat worker; this module cannot place orders."""
    worker_id = _safe_worker_id(worker_id)
    if not token or len(token) < 32 or generation < 1:
        raise ValueError("worker_credential_or_generation_invalid")
    if (
        not base_url.startswith("http://control-plane:")
        and not base_url.startswith("http://127.0.0.1:")
        and not base_url.startswith("http://localhost:")
    ):
        raise ValueError("worker_control_plane_url_not_allowed")
    path = Path(database_path)
    _initialize_worker_state(path, worker_id, generation)
    stop_event = threading.Event()

    def _request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop_event.set()

    previous_term = signal.signal(signal.SIGTERM, _request_stop)
    previous_int = signal.signal(signal.SIGINT, _request_stop)
    failures = 0
    completed = 0
    try:
        while not stop_event.is_set() and (
            iterations is None or completed < iterations
        ):
            try:
                heartbeat_sender(base_url, worker_id, token, generation)
                _record_heartbeat(path, generation)
                failures = 0
            except WorkerAuthorizationFailure:
                return 2
            except (OSError, TimeoutError):
                failures += 1
                if failures >= 3:
                    return 3
            completed += 1
            if iterations is None:
                stop_event.wait(interval_seconds)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    return 0


def main() -> int:
    worker_id = os.environ.get("ALIx_WORKER_ID", "")
    token = os.environ.get("ALIx_WORKER_TOKEN", "")
    base_url = os.environ.get("ALIx_CONTROL_PLANE_URL", "http://control-plane:8080")
    generation_raw = os.environ.get("ALIx_WORKER_GENERATION", "")
    database_path = os.environ.get("ALIx_WORKER_DB_PATH", "/worker-state/worker.sqlite")
    if not generation_raw.isdigit():
        print("worker_start_rejected: invalid_generation", file=sys.stderr)
        return 2
    try:
        return run_worker(
            worker_id,
            token,
            int(generation_raw),
            base_url,
            database_path,
        )
    except (ValueError, WorkerRuntimeFailure):
        print("worker_start_rejected: invalid_identity_or_state", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
