"""Capture immutable public Hugging Face model and tokenizer pins in GitHub Actions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
import httpx

MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "role": "kronos_small",
        "repository": "NeoQuasar/Kronos-small",
        "weight_file": "model.safetensors",
    },
    {
        "role": "kronos_tokenizer",
        "repository": "NeoQuasar/Kronos-Tokenizer-base",
        "weight_file": "model.safetensors",
    },
    {
        "role": "chronos_2_small",
        "repository": "autogluon/chronos-2-small",
        "weight_file": "model.safetensors",
    },
)
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


class ModelPinError(RuntimeError):
    """Raised when the Hub does not return a verifiable immutable model pin."""


def _verify_secure_response(response: httpx.Response, repository: str) -> None:
    for hop in (*response.history, response):
        if hop.url.scheme != "https":
            raise ModelPinError(f"hub_api_non_https_redirect:{repository}")
        host = hop.url.host or ""
        if host != "huggingface.co" and not host.endswith(".huggingface.co"):
            raise ModelPinError(f"hub_api_redirect_host_rejected:{repository}")


def _fetch_repo_metadata(repository: str) -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/{repository}/revision/main?blobs=true"
    try:
        with httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            max_redirects=3,
        ) as client:
            response = client.get(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "AliX-FourArm-ModelPinCapture/1.0",
                },
            )
            _verify_secure_response(response, repository)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ModelPinError(
            f"hub_api_request_failed:{repository}:{type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelPinError(f"hub_api_payload_invalid:{repository}")
    return payload


def _verify_weight_digest(
    repository: str,
    weight_file: str,
    siblings: Any,
) -> tuple[str, int]:
    if not isinstance(siblings, list):
        raise ModelPinError(f"model_files_missing:{repository}")
    weight_entry = next(
        (
            entry
            for entry in siblings
            if isinstance(entry, dict) and entry.get("rfilename") == weight_file
        ),
        None,
    )
    if not isinstance(weight_entry, dict):
        raise ModelPinError(f"weight_file_missing:{repository}:{weight_file}")
    lfs_info = weight_entry.get("lfs")
    if not isinstance(lfs_info, dict):
        raise ModelPinError(f"weight_lfs_digest_missing:{repository}:{weight_file}")
    raw_digest = lfs_info.get("oid", lfs_info.get("sha256"))
    if not isinstance(raw_digest, str):
        raise ModelPinError(
            f"weight_sha256_missing:{repository}:{weight_file}:"
            f"lfs_keys={sorted(lfs_info.keys())}"
        )
    weight_sha256 = raw_digest.removeprefix("sha256:")
    if not _SHA64.fullmatch(weight_sha256):
        raise ModelPinError(
            f"weight_sha256_invalid:{repository}:{weight_file}:"
            f"lfs_oid_format={raw_digest[:24]}"
        )
    size = lfs_info.get("size", weight_entry.get("size"))
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ModelPinError(f"weight_size_invalid:{repository}:{weight_file}")
    return weight_sha256, size


def _build_pin_record(
    spec: dict[str, str],
    info: dict[str, Any],
) -> dict[str, Any]:
    repository = spec["repository"]
    weight_file = spec["weight_file"]
    revision = info.get("sha")
    if not isinstance(revision, str) or not _SHA40.fullmatch(revision):
        raise ModelPinError(f"immutable_revision_missing:{repository}")
    weight_sha256, size = _verify_weight_digest(
        repository,
        weight_file,
        info.get("siblings"),
    )
    card_data = info.get("cardData")
    license_name = card_data.get("license") if isinstance(card_data, dict) else None
    return {
        "role": spec["role"],
        "repository": repository,
        "revision": revision,
        "revision_requested": "main",
        "weight_file": weight_file,
        "weights_sha256": weight_sha256,
        "weight_size_bytes": size,
        "license": license_name if isinstance(license_name, str) else "UNKNOWN",
        "hub_api_url": (
            f"https://huggingface.co/api/models/{repository}/revision/main?blobs=true"
        ),
    }


def capture_pin(spec: dict[str, str]) -> dict[str, Any]:
    return _build_pin_record(spec, _fetch_repo_metadata(spec["repository"]))


def capture_all() -> dict[str, Any]:
    captured_at = datetime.now(timezone.utc).isoformat()
    pins = [capture_pin(spec) for spec in MODEL_SPECS]
    return {
        "schema_version": "alix.four-arm.model-pins.v1",
        "captured_at_utc": captured_at,
        "source": "Hugging Face Hub model revision API (?blobs=true)",
        "pins": pins,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    manifest = capture_all()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    print("FOUR_ARM_MODEL_PIN_CAPTURE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
