"""Trusted host-side Docker Compose dispatcher; never runs inside the API container."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_entry(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = _unquote_env_value(value.strip())
    if not key or not key.replace("_", "").isalnum():
        return None
    return key, value


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without printing any values."""
    if not path.is_file():
        raise FileNotFoundError("env_file_not_found")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        entry = _parse_env_entry(raw_line)
        if entry is not None:
            key, value = entry
            os.environ.setdefault(key, value)


def _is_loopback_hostname(hostname: str | None) -> bool:
    return hostname in {"127.0.0.1", "localhost"}


def _url_has_no_credentials_or_suffix(parsed: Any) -> bool:
    return (
        parsed.username is None
        and parsed.password is None
        and parsed.path in ("", "/")
        and not parsed.query
        and not parsed.fragment
    )


def _valid_optional_port(port: int | None) -> bool:
    return port is None or 1 <= port <= 65535


def _local_control_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    try:
        port = parsed.port
    except ValueError:
        raise RuntimeError("control_plane_url_not_allowed") from None
    if (
        parsed.scheme != "http"
        or not _is_loopback_hostname(parsed.hostname)
        or not _url_has_no_credentials_or_suffix(parsed)
        or not _valid_optional_port(port)
    ):
        raise RuntimeError("control_plane_url_not_allowed")
    return base_url.rstrip("/")


def _api_request(
    base_url: str,
    admin_token: str,
    path: str,
    method: str = "POST",
    payload: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> Any:
    safe_base_url = _local_control_base_url(base_url)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{safe_base_url}{path}",
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {admin_token}",
            "Content-Type": "application/json",
        },
    )
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), urllib.request.HTTPHandler()
        )
        with opener.open(request, timeout=timeout) as response:
            if response.status == 204:
                return None
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"control_plane_http_{exc.code}") from None
    except urllib.error.URLError:
        raise RuntimeError("control_plane_unavailable") from None


def _docker(
    args: list[str],
    timeout: float = 30.0,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("docker_cli_unavailable_or_timeout") from None


def _container_info(container_name: str) -> dict[str, Any] | None:
    result = _docker(["container", "inspect", container_name])
    if result.returncode != 0:
        return None
    try:
        raw = json.loads(result.stdout)
        if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
            raise ValueError("invalid_docker_inspect")
        return raw[0]
    except (json.JSONDecodeError, ValueError, TypeError):
        raise RuntimeError("docker_inspect_invalid") from None


def _container_name(worker_id: str, generation: int) -> str:
    clean_id = "".join(ch for ch in worker_id.lower() if ch.isalnum())[:12]
    if not clean_id or generation < 0:
        raise ValueError("invalid_worker_container_identity")
    return f"alix-worker-{clean_id}-g{generation}"


def _start_worker(
    command: dict[str, Any],
    base_url: str,
    token: str,
    compose_file: Path,
    project_name: str,
) -> str:
    worker_id = str(command["worker_id"])
    generation = int(command["generation"])
    if generation < 1:
        raise ValueError("invalid_worker_generation")
    name = _container_name(worker_id, generation)
    existing = _container_info(name)
    if existing is not None:
        labels = existing.get("Config", {}).get("Labels", {})
        if labels.get("alix.worker.id") != worker_id or labels.get(
            "alix.worker.generation"
        ) != str(generation):
            raise RuntimeError("container_identity_mismatch")
        if existing.get("State", {}).get("Running") is True:
            return name
        removed = _docker(["container", "rm", name])
        if removed.returncode != 0:
            raise RuntimeError("stale_container_remove_failed")
    worker_token = secrets.token_urlsafe(40)
    digest = hashlib.sha256(worker_token.encode("utf-8")).hexdigest()
    _api_request(
        base_url,
        token,
        f"/v1/admin/workers/{worker_id}/runtime-credential",
        method="PUT",
        payload={"generation": generation, "credential_sha256": digest},
    )
    volume_name = f"alix-worker-state-{''.join(ch for ch in worker_id.lower() if ch.isalnum())[:12]}-g{generation}"
    worker_environment = dict(os.environ)
    worker_environment["ALIx_WORKER_TOKEN"] = worker_token
    result = _docker(
        [
            "compose",
            "--project-name",
            project_name,
            "-f",
            str(compose_file),
            "run",
            "--detach",
            "--no-deps",
            "--name",
            name,
            "--volume",
            f"{volume_name}:/worker-state",
            "--label",
            f"alix.worker.id={worker_id}",
            "--label",
            f"alix.worker.generation={generation}",
            "--env",
            f"ALIx_WORKER_ID={worker_id}",
            "--env",
            "ALIx_WORKER_TOKEN",
            "--env",
            f"ALIx_WORKER_GENERATION={generation}",
            "worker",
        ],
        timeout=90.0,
        environment=worker_environment,
    )
    if result.returncode != 0:
        raise RuntimeError("docker_worker_start_failed")
    return name


def process_command(
    command: dict[str, Any],
    base_url: str,
    admin_token: str,
    compose_file: Path,
    project_name: str,
    dispatcher_id: str,
) -> None:
    action = str(command["action"])
    worker_id = str(command["worker_id"])
    generation = int(command["generation"])
    if action == "START":
        container_name = _start_worker(
            command, base_url, admin_token, compose_file, project_name
        )
        success = True
        error_code = None
    elif action in ("STOP", "DISABLE"):
        container_name = _container_name(worker_id, generation)
        info = _container_info(container_name)
        success = True
        error_code = None
        if info is not None:
            labels = info.get("Config", {}).get("Labels", {})
            if labels.get("alix.worker.id") != worker_id or labels.get(
                "alix.worker.generation"
            ) != str(generation):
                success = False
                error_code = "container_identity_mismatch"
            elif info.get("State", {}).get("Running") is True:
                stopped = _docker(["stop", "--time", "10", container_name])
                success = stopped.returncode == 0
                if not success:
                    error_code = "docker_worker_stop_failed"
    else:
        raise RuntimeError("unknown_runtime_command")
    _api_request(
        base_url,
        admin_token,
        f"/v1/admin/runtime-commands/{command['command_id']}/ack",
        payload={
            "success": success,
            "container_name": container_name if action == "START" and success else None,
            "error_code": error_code,
        },
    )


def dispatch_once(
    base_url: str,
    admin_token: str,
    compose_file: Path,
    project_name: str,
    dispatcher_id: str,
    lease_seconds: int = 30,
    limit: int = 10,
) -> int:
    commands = _api_request(
        base_url,
        admin_token,
        "/v1/admin/runtime-commands/claim",
        payload={
            "dispatcher_id": dispatcher_id,
            "lease_seconds": lease_seconds,
            "limit": limit,
        },
    )
    if not isinstance(commands, list):
        raise RuntimeError("runtime_command_response_invalid")
    processed = 0
    for command in commands:
        if not isinstance(command, dict):
            raise RuntimeError("runtime_command_invalid")
        try:
            process_command(
                command,
                base_url,
                admin_token,
                compose_file,
                project_name,
                dispatcher_id,
            )
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            code = (
                str(exc)
                if str(exc).startswith(
                    ("docker_", "container_", "control_plane_", "unknown_")
                )
                else "runtime_dispatch_failed"
            )
            _api_request(
                base_url,
                admin_token,
                f"/v1/admin/runtime-commands/{command.get('command_id', '')}/ack",
                payload={"success": False, "container_name": None, "error_code": code},
            )
        processed += 1
    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--compose-file", default="compose.yaml")
    parser.add_argument("--project-name", default="alix")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    arguments = parser.parse_args()
    load_env_file(Path(arguments.env_file))
    token = os.environ.get("ALIx_ADMIN_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("ALIx_ADMIN_TOKEN must contain at least 32 characters")
    base_url = os.environ.get("ALIx_HOST_CONTROL_URL", "http://127.0.0.1:8080")
    compose_file = Path(arguments.compose_file).resolve()
    dispatcher_id = f"host-{os.getpid()}"
    while True:
        dispatch_once(
            base_url, token, compose_file, arguments.project_name, dispatcher_id
        )
        if arguments.once:
            return 0
        time.sleep(max(0.5, arguments.interval))


if __name__ == "__main__":
    raise SystemExit(main())
