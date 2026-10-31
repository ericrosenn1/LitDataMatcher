"""Serial in-process model reuse and durable native-analysis batches."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .controller import OwnershipConflict
from .data_plane import digest
from .workflow_checkpoint import analysis_fingerprint, file_identity


def runtime_configuration(device, overrides=None):
    """Resolve concrete defaults without fingerprinting mutable backend objects."""
    from .semantic_runtime import RuntimeConfig

    values = asdict(overrides) if is_dataclass(overrides) else dict(overrides or {})
    if "device" in values and values["device"] != device:
        raise ValueError("Runtime configuration device disagrees with analysis device")
    values.setdefault("device", device)
    values.setdefault("dtype", "bfloat16" if device == "cuda" else "float32")
    return asdict(RuntimeConfig(**values))


def runtime_environment():
    """Bind optional decoder versions even before a lazy backend is constructed."""
    from .semantic_runtime import DEPENDENCY_IDENTITY_VERSION, runtime_dependency_versions

    versions = runtime_dependency_versions()
    return {
        "lm_format_enforcer": versions["lm-format-enforcer"],
        "dependency_identity_version": DEPENDENCY_IDENTITY_VERSION,
        "dependency_versions": versions,
    }


def _identity_paths(value):
    if isinstance(value, dict):
        if "path" in value:
            yield Path(value["path"])
        for child in value.values():
            yield from _identity_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from _identity_paths(child)


def _stat_identity(path):
    if not path.exists():
        return None
    stat = path.stat()
    if path.is_dir():
        return (
            "directory",
            tuple(
                sorted(str(item.relative_to(path)) for item in path.rglob("*") if item.is_file())
            ),
        )
    return ("file", stat.st_size, stat.st_mtime_ns)


class AnalysisSession:
    """Load each backend once; bind it to one serial owner and configuration.

    Shared content is hashed at entry and explicit final validation. Lightweight
    stat checks protect every case boundary. Per-case documents are hashed anew.
    """

    def __init__(
        self,
        root,
        model_dir,
        embedding_dir,
        *,
        device="cpu",
        vector_cache=None,
        runtime_config=None,
        runtime_qualification=None,
    ):
        self.root, self.model_dir, self.embedding_dir = (
            Path(path).resolve() for path in (root, model_dir, embedding_dir)
        )
        self.device = device
        self.vector_cache = str(Path(vector_cache).resolve()) if vector_cache else None
        self.runtime_config = runtime_configuration(device, runtime_config)
        self.runtime_qualification = (
            str(Path(runtime_qualification).resolve()) if runtime_qualification else None
        )
        if self.runtime_qualification and not Path(self.runtime_qualification).is_file():
            raise ValueError("Explicit runtime qualification must be an existing file")
        self._base = analysis_fingerprint(self.root, self.model_dir, self.embedding_dir, {})
        self._base["runtime_environment"] = runtime_environment()
        self._base["qualification_override"] = (
            file_identity(self.runtime_qualification) if self.runtime_qualification else None
        )
        self._stats = {path: _stat_identity(path) for path in _identity_paths(self._base)}
        self._owner = threading.get_ident()
        self._process = os.getpid()
        self._configuration_digest = self._config_digest()
        self._lock = threading.Lock()
        self._runtime = self._retriever = None
        self._retrieval_records_digest = None
        self._closed = False

    def _config_digest(self):
        return digest(
            {
                "root": str(self.root),
                "model": str(self.model_dir),
                "embeddings": str(self.embedding_dir),
                "device": self.device,
                "vector_cache": self.vector_cache,
                "runtime_config": self.runtime_config,
                "runtime_qualification": self.runtime_qualification,
            }
        )

    def validate_unchanged(self, *, full=False):
        if self._closed:
            raise RuntimeError("Analysis session is closed")
        if self._config_digest() != self._configuration_digest:
            raise ValueError("Analysis session configuration changed")
        if any(_stat_identity(path) != expected for path, expected in self._stats.items()):
            raise ValueError("Session inputs, models or implementation changed")
        if runtime_environment() != self._base["runtime_environment"]:
            raise ValueError("Session decoder environment changed")
        if full:
            current = analysis_fingerprint(self.root, self.model_dir, self.embedding_dir, {})
            current["runtime_environment"] = runtime_environment()
            current["qualification_override"] = (
                file_identity(self.runtime_qualification) if self.runtime_qualification else None
            )
            if current != self._base:
                raise ValueError("Session content fingerprint changed")

    @contextmanager
    def claim(self, root, model_dir, embedding_dir, parameters):
        if (
            os.getpid() != self._process
            or threading.get_ident() != self._owner
            or not self._lock.acquire(blocking=False)
        ):
            raise OwnershipConflict("Analysis session requires its original serial owner")
        try:
            actual = tuple(str(Path(path).resolve()) for path in (root, model_dir, embedding_dir))
            expected = tuple(str(path) for path in (self.root, self.model_dir, self.embedding_dir))
            if (
                actual != expected
                or parameters.get("device") != self.device
                or parameters.get("vector_cache") != self.vector_cache
                or parameters.get("runtime_config") != self.runtime_config
                or parameters.get("runtime_qualification") != self.runtime_qualification
            ):
                raise ValueError("Analysis session configuration identity mismatch")
            self.validate_unchanged()
            yield self
            self.validate_unchanged()
        finally:
            self._lock.release()

    def fingerprint(self, parameters):
        self.validate_unchanged()
        value = deepcopy(self._base)
        value["parameters"] = deepcopy(parameters)
        value["document"] = (
            file_identity(parameters["document_path"]) if parameters.get("document_path") else None
        )
        return value

    @property
    def runtime(self):
        if (
            os.getpid() != self._process
            or threading.get_ident() != self._owner
            or not self._lock.locked()
        ):
            raise OwnershipConflict("Runtime access requires an active serial session claim")
        if self._runtime is None:
            from .semantic_runtime import LocalSemanticRuntime, RuntimeConfig

            self._runtime = LocalSemanticRuntime(
                self.model_dir, RuntimeConfig(**self.runtime_config)
            )
        return self._runtime

    def retriever(self, records):
        if (
            os.getpid() != self._process
            or threading.get_ident() != self._owner
            or not self._lock.locked()
        ):
            raise OwnershipConflict("Retriever access requires an active serial session claim")
        if self._retriever is None:
            if self.vector_cache:
                from .retrieval import CachedSemanticIndex

                self._retriever = CachedSemanticIndex(
                    self.embedding_dir, self.vector_cache, device=self.device
                )
            else:
                from .semantic_runtime import PretrainedSemanticIndex

                self._retriever = PretrainedSemanticIndex(self.embedding_dir, device=self.device)
        identity = digest(records)
        if identity != self._retrieval_records_digest:
            self._retriever.fit(records)
            self._retrieval_records_digest = identity
        return self._retriever

    def close(self, *, validate=True):
        if validate:
            self.validate_unchanged(full=True)
        self._runtime = self._retriever = None
        self._closed = True
