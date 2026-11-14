"""Acquire commit-pinned public model files and verify every artifact digest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import quote

import httpx

DEFAULT_PIN_PATH = Path("config/four_arm/model_pins.json")
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")
_METADATA_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    }
)


class ModelAcquisitionError(RuntimeError):
    """Raised when a pinned model artifact cannot be acquired and verified."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_digest(actual_sha256: str, expected_sha256: str) -> None:
    if not _SHA64.fullmatch(expected_sha256) or actual_sha256 != expected_sha256:
        raise ModelAcquisitionError("model_weight_sha256_mismatch")


def _safe_filename(filename: str) -> str:
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or ".." in filename
    ):
        raise ModelAcquisitionError("model_asset_filename_invalid")
    return filename


def _build_download_url(repository: str, revision: str, filename: str) -> str:
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
        or not _SHA40.fullmatch(revision)
    ):
        raise ModelAcquisitionError("model_repository_or_revision_invalid")
    safe_name = _safe_filename(filename)
    return (
        f"https://huggingface.co/{repository}/resolve/{revision}/"
        f"{quote(safe_name, safe='')}"
    )


def _verify_https_redirect_chain(response: httpx.Response) -> None:
    for hop in (*response.history, response):
        host = hop.url.host or ""
        if hop.url.scheme != "https":
            raise ModelAcquisitionError("model_download_non_https_redirect")
        if (
            host != "huggingface.co"
            and not host.endswith(".huggingface.co")
            and not host.endswith(".hf.co")
        ):
            raise ModelAcquisitionError("model_download_redirect_host_rejected")


def _read_pins(
    path: Path,
) -> tuple[dict[str, Any], list[tuple[str, Mapping[str, Any]]]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ModelAcquisitionError("model_pin_manifest_unreadable") from None
    if not isinstance(raw, dict) or raw.get("schema_version") != "alix.four-arm.model-pins.v1":
        raise ModelAcquisitionError("model_pin_manifest_schema_invalid")
    models = raw.get("models")
    if not isinstance(models, Mapping):
        raise ModelAcquisitionError("model_pin_manifest_models_invalid")
    kronos = models.get("kronos_small")
    chronos = models.get("chronos_2")
    if not isinstance(kronos, Mapping) or not isinstance(chronos, Mapping):
        raise ModelAcquisitionError("model_pin_manifest_models_invalid")
    tokenizer = kronos.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise ModelAcquisitionError("model_pin_tokenizer_missing")
    groups = [
        ("kronos_small", kronos),
        ("kronos_tokenizer", tokenizer),
        ("chronos_2", chronos),
    ]
    for group, record in groups:
        repository = record.get("repository")
        revision = record.get("revision")
        weight_file = record.get("weight_file")
        expected_sha = record.get("weights_sha256")
        if (
            not isinstance(repository, str)
            or not isinstance(revision, str)
            or not isinstance(weight_file, str)
            or not isinstance(expected_sha, str)
        ):
            raise ModelAcquisitionError(f"model_pin_fields_invalid:{group}")
        _build_download_url(repository, revision, weight_file)
        if not _SHA64.fullmatch(expected_sha):
            raise ModelAcquisitionError(f"model_pin_sha256_invalid:{group}")
    return raw, groups


def _fetch_repo_files(
    client: httpx.Client,
    repository: str,
    revision: str,
    weight_file: str,
) -> list[str]:
    url = f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true"
    try:
        response = client.get(url, headers={"Accept": "application/json"})
        _verify_https_redirect_chain(response)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ModelAcquisitionError(
            f"model_revision_metadata_failed:{repository}:{type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("sha") != revision:
        raise ModelAcquisitionError(f"model_revision_mismatch:{repository}")
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise ModelAcquisitionError(f"model_revision_files_invalid:{repository}")
    filenames = [
        entry["rfilename"]
        for entry in siblings
        if isinstance(entry, dict)
        and isinstance(entry.get("rfilename"), str)
        and (
            entry.get("rfilename") in _METADATA_FILES
            or entry.get("rfilename") == weight_file
        )
    ]
    if weight_file not in filenames:
        raise ModelAcquisitionError(f"pinned_weight_file_missing:{repository}")
    return sorted(set(_safe_filename(name) for name in filenames))


def _download_asset(
    client: httpx.Client,
    *,
    group: str,
    repository: str,
    revision: str,
    filename: str,
    destination_root: Path,
    expected_sha256: str | None,
    expected_size: int | None,
) -> dict[str, Any]:
    url = _build_download_url(repository, revision, filename)
    destination = destination_root / group / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    digest = sha256()
    size = 0
    try:
        with client.stream("GET", url) as response:
            _verify_https_redirect_chain(response)
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
    except (httpx.HTTPError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise ModelAcquisitionError(
            f"model_asset_download_failed:{group}:{filename}:{type(exc).__name__}"
        ) from exc
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None:
        try:
            _assert_digest(actual_sha256, expected_sha256)
        except ModelAcquisitionError:
            partial.unlink(missing_ok=True)
            raise
    if expected_size is not None and size != expected_size:
        partial.unlink(missing_ok=True)
        raise ModelAcquisitionError(f"model_asset_size_mismatch:{group}:{filename}")
    partial.replace(destination)
    return {
        "group": group,
        "repository": repository,
        "revision": revision,
        "filename": filename,
        "path": f"{group}/{filename}",
        "sha256": actual_sha256,
        "size_bytes": size,
        "pinned_weight": expected_sha256 is not None,
        "source_url": url,
    }


def acquire_models(pin_path: Path, output_dir: Path) -> dict[str, Any]:
    raw_pins, groups = _read_pins(pin_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
    with httpx.Client(timeout=timeout, follow_redirects=True, max_redirects=3) as client:
        for group, pin in groups:
            repository = str(pin["repository"])
            revision = str(pin["revision"])
            weight_file = str(pin["weight_file"])
            filenames = _fetch_repo_files(client, repository, revision, weight_file)
            if group in {"kronos_small", "chronos_2"} and "config.json" not in filenames:
                raise ModelAcquisitionError(f"model_config_missing:{group}")
            for filename in filenames:
                is_weight = filename == weight_file
                records.append(
                    _download_asset(
                        client,
                        group=group,
                        repository=repository,
                        revision=revision,
                        filename=filename,
                        destination_root=output_dir,
                        expected_sha256=str(pin["weights_sha256"]) if is_weight else None,
                        expected_size=(
                            int(pin["weight_size_bytes"])
                            if is_weight and isinstance(pin.get("weight_size_bytes"), int)
                            else None
                        ),
                    )
                )
    manifest = {
        "schema_version": "alix.four-arm.model-artifacts.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pins_manifest_sha256": _sha256_file(pin_path),
        "files": records,
        "model_groups": [group for group, _ in groups],
        "safety": {
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "posthack_holdout_access": False,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    print("FOUR_ARM_MODEL_ACQUISITION=PASS")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-pins", type=Path, default=DEFAULT_PIN_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    acquire_models(arguments.model_pins, arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
