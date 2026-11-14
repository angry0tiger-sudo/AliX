"""Mock-only tests for the pinned model artifact acquisition helpers."""

from __future__ import annotations

from hashlib import sha256

import pytest

from scripts.acquire_four_arm_models import (
    ModelAcquisitionError,
    _assert_digest,
    _build_download_url,
    _safe_filename,
)


def test_download_url_uses_immutable_commit_revision() -> None:
    url = _build_download_url(
        "NeoQuasar/Kronos-small",
        "901c26c1332695a2a8f243eb2f37243a37bea320",
        "model.safetensors",
    )
    assert url == (
        "https://huggingface.co/NeoQuasar/Kronos-small/resolve/"
        "901c26c1332695a2a8f243eb2f37243a37bea320/model.safetensors"
    )


@pytest.mark.parametrize(
    "filename", ["../model.safetensors", "/tmp/model", "a/b", "x..y"]
)
def test_asset_filename_path_traversal_is_rejected(filename: str) -> None:
    with pytest.raises(ModelAcquisitionError, match="model_asset_filename_invalid"):
        _safe_filename(filename)


def test_weight_digest_mismatch_fails_closed() -> None:
    expected = sha256(b"pinned bytes").hexdigest()
    actual = sha256(b"different bytes").hexdigest()
    with pytest.raises(ModelAcquisitionError, match="model_weight_sha256_mismatch"):
        _assert_digest(actual, expected)
