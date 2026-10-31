"""Small, leased checkpoints for the native analysis stages.

Only derived artifacts live here. Scientific input, model and implementation
fingerprints must match before an interrupted run can resume.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import platform
import sys
import time
from pathlib import Path

from .controller import integration_lease
from .data_plane import atomic_json, digest
from .strict_artifacts import strict_json_loads, validate_artifact


def initial_manifest(run_id, started_at):
    """Schema-valid running state; unmeasured values remain null or empty."""
    return {
        "schema_version": "2.0",
        "run_id": run_id,
        "execution_status": "RUNNING",
        "started_at": dt.datetime.fromtimestamp(started_at, dt.timezone.utc).isoformat(),
        "finished_at": None,
        "source": {
            "repository": "ericrosenn1/LitDataMatcher",
            "commit": None,
            "working_tree_digest": None,
            "spec_digest": None,
            "config_digest": None,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependency_lock_digest": None,
            "hardware_record": None,
        },
        "models": [],
        "source_snapshots": [],
        "commands": [],
        "evaluation": {
            "protocol_version": None,
            "split_id": None,
            "split_role": "UNASSIGNED",
            "label_origins": [],
            "holdout_exposed_to_tuning": False,
        },
        "coverage": {
            key: 0
            for key in (
                "unique_literature_records",
                "parsed_full_texts",
                "unique_accession_studies",
                "sample_profiled_studies",
                "inspected_processed_studies",
                "external_structured_resources",
                "distinct_pilot_contexts",
                "case_dossiers",
            )
        },
        "artifacts": [],
        "network": {
            "mode": "UNSET",
            "offline_block_test": False,
            "external_requests_observed": None,
        },
        "inference": {"fresh_calls": 0, "cache_replays": 0, "backend_qualified": False},
        "failures": [],
    }


def file_identity(path):
    path = Path(path).resolve()
    if not path.exists():
        return {"path": str(path), "missing": True}
    if path.is_dir():
        return {
            "path": str(path),
            "files": [file_identity(p) for p in sorted(path.rglob("*")) if p.is_file()],
        }
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return {"path": str(path), "sha256": checksum.hexdigest(), "size_bytes": path.stat().st_size}


def analysis_fingerprint(root, model_dir, embedding_dir, parameters):
    from .semantic_runtime import runtime_dependency_versions

    root = Path(root).resolve()
    package = Path(__file__).parent
    versions = runtime_dependency_versions()
    return {
        "parameters": parameters,
        "inputs": [
            file_identity(root / path)
            for path in (
                "catalog/literature.jsonl",
                "catalog/studies.jsonl",
                "catalog/processed_inspections.jsonl",
                "runtime-qualification/qualified_7b_pass1.json",
                "external_evidence/uniprot",
            )
        ],
        "document": file_identity(parameters["document_path"])
        if parameters.get("document_path")
        else None,
        "models": [file_identity(model_dir), file_identity(embedding_dir)],
        "implementation": [file_identity(p) for p in sorted(package.glob("*.py"))]
        + [
            file_identity(package / "question_compiler_rules_v1.json"),
            file_identity(package / "schemas_v2"),
        ],
        "libraries": versions,
    }


class CheckpointRun:
    def __init__(self, out: Path, fingerprint: dict, resume: bool):
        self.out = Path(out).resolve()
        self.directory = self.out / ".workflow"
        self.path = self.directory / "state.json"
        self.fingerprint = fingerprint
        self.resume = resume

    def __enter__(self):
        existed = self.out.exists()
        if existed and not self.resume:
            raise ValueError("Run output already exists; use --resume with the original parameters")
        if not existed and self.resume:
            raise ValueError("Cannot resume a missing run directory")
        if self.resume and not self.path.is_file():
            raise ValueError(
                "Run has no resumable checkpoint; preserve it and choose a new output directory"
            )
        self.lease = integration_lease(self.directory, "native-analysis")
        self.lease.__enter__()
        try:
            if not self.resume and (
                self.path.exists() or any(path.name != ".workflow" for path in self.out.iterdir())
            ):
                raise ValueError(
                    "Run output was created by another owner; preserve it and use --resume"
                )
            if self.resume:
                if not self.path.is_file():
                    raise ValueError(
                        "Run has no resumable checkpoint; preserve it and choose a new output directory"
                    )
                self.state = strict_json_loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(self.state, dict) or not isinstance(
                    self.state.get("stages"), dict
                ):
                    raise ValueError(
                        "Invalid checkpoint journal; preserve this run and use a new output directory"
                    )
                if self.state.get("fingerprint") != digest(self.fingerprint):
                    raise ValueError(
                        "Resume fingerprint mismatch: input, parameters, models or implementation changed"
                    )
            else:
                self.state = {
                    "schema_version": 1,
                    "fingerprint": digest(self.fingerprint),
                    "inputs": self.fingerprint,
                    "started_at": time.time(),
                    "stages": {},
                    "repairs": [],
                }
                atomic_json(
                    self.out / "RUN_MANIFEST.json",
                    initial_manifest(self.out.name, self.state["started_at"]),
                )
            self.state["status"] = "RUNNING"
            self._write()
            return self
        except BaseException:
            self.lease.__exit__(*sys.exc_info())
            raise

    def _write(self):
        self.state["updated_at"] = time.time()
        atomic_json(self.path, self.state)

    def _parents(self, parents):
        return {name: self.state["stages"].get(name, {}).get("sha256") for name in parents}

    def load(self, name, parents=()):
        receipt = self.state["stages"].get(name)
        if not receipt:
            return None
        path = (self.directory / receipt["file"]).resolve()
        if path.parent != self.directory.resolve() or path.suffix != ".json":
            raise ValueError("Checkpoint path is outside this run")
        try:
            if receipt.get("parents", {}) != self._parents(parents):
                raise ValueError("Stage dependency changed")
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
                raise ValueError("Stage checksum mismatch")
            return strict_json_loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError) as exc:
            if path.is_file():
                quarantine = self.directory / "quarantine" / f"{time.time_ns()}-{path.name}"
                quarantine.parent.mkdir(exist_ok=True)
                path.replace(quarantine)
            self.state["repairs"].append({"stage": name, "reason": str(exc)})
            self.state["stages"].pop(name, None)
            self._write()
            return None

    def save(self, name, value, parents=()):
        path = self.directory / (hashlib.sha256(name.encode()).hexdigest() + ".json")
        atomic_json(path, value)
        self.state["stages"][name] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parents": self._parents(parents),
        }
        self._write()
        return value

    def cached(self, name, function, parents=()):
        value = self.load(name, parents)
        return value if value is not None else self.save(name, function(), parents)

    def completed(self):
        complete = self.state.get("completed")
        if not complete:
            return None
        try:
            for receipt in complete["outputs"]:
                path = (self.out / receipt["file"]).resolve()
                if path.parent != self.out:
                    raise ValueError("Completed artifact is outside this run")
                if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
                    raise ValueError(f"Output changed: {path.name}")
                validate_artifact(path)
        except (OSError, ValueError) as exc:
            self.state["repairs"].append({"stage": "publication", "reason": str(exc)})
            self.state.pop("completed", None)
            self._write()
            return None
        return complete["result"]

    def finish(self, result):
        if self.state.get("retryable_stage_failures") or result.get("status") == "FAIL":
            self.state["status"] = "FAILED" if result.get("status") == "FAIL" else "PARTIAL"
            self.state.pop("completed", None)
            self._write()
            return
        self.state["completed"] = {
            "result": result,
            "outputs": [
                {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(self.out.iterdir())
                if p.is_file()
            ],
        }
        self.state["status"] = "COMPLETE"
        self._write()

    def __exit__(self, kind, value, traceback):
        try:
            if kind:
                self.state["status"] = (
                    "INTERRUPTED" if issubclass(kind, (KeyboardInterrupt, SystemExit)) else "FAILED"
                )
                self.state["error"] = {"type": kind.__name__, "description": str(value)}
                path = self.out / "RUN_MANIFEST.json"
                try:
                    manifest = strict_json_loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    manifest = initial_manifest(self.out.name, self.state["started_at"])
                manifest["execution_status"] = (
                    "INTERRUPTED" if self.state["status"] == "INTERRUPTED" else "FAIL"
                )
                manifest["failures"].append({"stage": "workflow", "description": str(value)})
                atomic_json(path, manifest)
            elif self.state.get("completed"):
                self.state["status"] = "COMPLETE"
                self.state.pop("error", None)
            self._write()
        finally:
            self.lease.__exit__(kind, value, traceback)
