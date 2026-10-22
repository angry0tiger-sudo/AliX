"""Hash-verified local Hugging Face loading with no download fallback."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from services.nlp_inference.service import ModelManifest, NlpServiceError


class Pipeline(Protocol):
    def __call__(self, text: str, **kwargs: object) -> object:
        """Run local text classification."""


class LocalHuggingFaceClassifier:
    """A minimal adapter that normalizes the local pipeline's result shape."""

    def __init__(
        self, pipeline: Pipeline, label_map: tuple[tuple[str, str], ...]
    ) -> None:
        self._pipeline = pipeline
        self._label_map = dict(label_map)

    def predict(self, text: str) -> Mapping[str, object]:
        try:
            output = self._pipeline(text, truncation=True, top_k=1)
        except TimeoutError:
            raise NlpServiceError("nlp_timeout") from None
        except Exception:
            raise NlpServiceError("nlp_local_pipeline_failed") from None
        if isinstance(output, list) and output and isinstance(output[0], list):
            output = output[0]
        if isinstance(output, list) and output and isinstance(output[0], Mapping):
            output = output[0]
        if not isinstance(output, Mapping):
            raise NlpServiceError("nlp_local_pipeline_output_invalid")
        label = output.get("label")
        score = output.get("score")
        if not isinstance(label, str) or label not in self._label_map:
            raise NlpServiceError("nlp_local_pipeline_label_invalid")
        return {"label": label, "score": score}


def _verify_artifacts(manifest: ModelManifest) -> Path:
    if manifest.test_fixture:
        raise NlpServiceError("nlp_test_fixture_cannot_load_runtime_model")
    try:
        root = Path(manifest.artifact_dir).expanduser().resolve(strict=True)
    except OSError:
        raise NlpServiceError("nlp_artifact_directory_missing") from None
    if not root.is_dir():
        raise NlpServiceError("nlp_artifact_directory_missing")
    for relative, expected in (
        (manifest.weights_file, manifest.weights_sha256),
        (manifest.tokenizer_file, manifest.tokenizer_sha256),
    ):
        try:
            path = (root / relative).resolve(strict=True)
        except OSError:
            raise NlpServiceError("nlp_artifact_file_missing") from None
        if root not in path.parents or not path.is_file():
            raise NlpServiceError("nlp_artifact_path_escape")
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            raise NlpServiceError("nlp_artifact_file_unreadable") from None
        if actual != expected:
            raise NlpServiceError("nlp_artifact_hash_mismatch")
    return root


def load_local_huggingface_classifier(
    manifest: ModelManifest,
    *,
    pipeline_factory: Callable[[Path], object] | None = None,
) -> LocalHuggingFaceClassifier:
    """Load from verified local files only; never resolve model ID remotely."""
    root = _verify_artifacts(manifest)
    if pipeline_factory is not None:
        loaded = pipeline_factory(root)
    else:
        try:
            transformers = cast(Any, import_module("transformers"))
        except ImportError:
            raise NlpServiceError("nlp_transformers_unavailable") from None
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(root), local_files_only=True, trust_remote_code=False
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                str(root),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            loaded = transformers.pipeline(
                "text-classification", model=model, tokenizer=tokenizer, device=-1
            )
        except Exception:
            raise NlpServiceError("nlp_local_model_load_failed") from None
    if not callable(loaded):
        raise NlpServiceError("nlp_local_pipeline_invalid")
    return LocalHuggingFaceClassifier(loaded, manifest.label_map)
