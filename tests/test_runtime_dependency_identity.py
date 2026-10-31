"""Dependency drift invalidates derived state without running model weights."""

import copy
import importlib.metadata
import json
from dataclasses import asdict

import pytest

from litdatamatcher.analysis_session import AnalysisSession
from litdatamatcher.retrieval_training import FIXED_CONFIG, training_fingerprint
from litdatamatcher.semantic_runtime import (
    RUNTIME_DEPENDENCIES,
    LocalSemanticRuntime,
    RuntimeConfig,
    digest,
    runtime_dependency_versions,
)
from litdatamatcher.v2 import qualified_runtime
from litdatamatcher.workflow_checkpoint import CheckpointRun, analysis_fingerprint


@pytest.fixture
def versions(monkeypatch):
    values = {name: "fixture-1" for name in RUNTIME_DEPENDENCIES}
    values["lm-format-enforcer"] = "0.11.3"
    original = importlib.metadata.version

    def lookup(name):
        if name not in values:
            return original(name)
        if values[name] is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return values[name]

    monkeypatch.setattr(importlib.metadata, "version", lookup)
    return values


def runtime():
    value = LocalSemanticRuntime.__new__(LocalSemanticRuntime)
    value.config = RuntimeConfig()
    value.model_manifest = {"model_id": "synthetic", "revision": "a" * 40, "license": "fixture"}
    return value


def source():
    return {"document_id": "synthetic:dependency", "text": "Treatment reduced IL6 in human cells."}


def payload():
    return {
        "claims": [
            {
                "quote": source()["text"],
                "subject": "Treatment",
                "predicate": "reduced",
                "object": "IL6",
                "direction": "decrease",
                "negated": False,
                "status": "direct_experiment",
                "context": None,
                "comparator": None,
            }
        ],
        "questions": [],
    }


def test_dependency_keys_are_stable_and_missing_optional_versions_are_null(versions):
    versions["interegular"] = None
    first = runtime_dependency_versions()
    assert list(first) == sorted(RUNTIME_DEPENDENCIES)
    assert first["interegular"] is None
    assert first == runtime_dependency_versions()
    assert json.loads(json.dumps(first)) == first


@pytest.mark.parametrize("name", RUNTIME_DEPENDENCIES)
def test_every_declared_dependency_changes_backend_and_inference_identity(versions, name):
    value = runtime()
    first = value.inference_fingerprint(source())
    versions[name] = "changed-version"
    second = value.inference_fingerprint(source())
    assert first["dependency_versions"][name] != second["dependency_versions"][name]
    assert first["input_sha256"] == second["input_sha256"]
    assert first["model_manifest_sha256"] == second["model_manifest_sha256"]
    assert digest(first) != digest(second)


def test_validator_drift_creates_new_inference_cache_without_overwriting_old(versions, tmp_path):
    value = runtime()
    calls = []

    def generate(*args):
        calls.append(True)
        return json.dumps(payload()), 10, 20

    value._generate = generate
    value.extract(source(), tmp_path)
    original = next(tmp_path.glob("*.json"))
    original_bytes = original.read_bytes()
    assert value.extract(source(), tmp_path)["inference_manifest"]["origin"] == "cache_replay"
    versions["jsonschema"] = "changed-validator"
    assert (
        value.extract(source(), tmp_path)["inference_manifest"]["origin"] == "fresh_local_inference"
    )
    assert len(calls) == 2 and len(list(tmp_path.glob("*.json"))) == 2
    assert original.read_bytes() == original_bytes


def test_qualification_rejects_dependency_drift_and_legacy_missing_identity(versions, tmp_path):
    value = runtime()
    value._generate = lambda *args: (json.dumps(payload()), 10, 20)
    fresh = value.extract(source())
    expected = fresh["inference_manifest"]["fingerprint"]
    report = {
        "qualification_schema": "runtime-qualification-v2.3",
        "status": "PASS",
        "backend_identity": value.backend_identity(),
        "expected_identity": expected,
        "configuration": asdict(value.config),
        "replay_identity": copy.deepcopy(expected),
        "replay_configuration": asdict(value.config),
        "replay_origin": "cache_replay",
        "network_control": {"blocked_probe": ["synthetic"]},
        "fresh": fresh,
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    original = path.read_bytes()
    assert qualified_runtime(path, value)[0]
    versions["tokenizers"] = "changed-tokenizer"
    assert not qualified_runtime(path, value)[0]
    assert path.read_bytes() == original
    versions["tokenizers"] = "fixture-1"
    for identity in (
        report["backend_identity"],
        report["expected_identity"],
        report["replay_identity"],
        report["fresh"]["inference_manifest"]["fingerprint"],
    ):
        identity.pop("dependency_versions", None)
        identity.pop("dependency_identity_version", None)
    path.write_text(json.dumps(report), encoding="utf-8")
    assert not qualified_runtime(path, value)[0]


def test_completed_workflow_and_open_session_reject_dependency_drift(versions, tmp_path):
    root, model, embeddings = (tmp_path / name for name in ("source", "model", "embeddings"))
    first = analysis_fingerprint(root, model, embeddings, {})
    output = tmp_path / "run"
    with CheckpointRun(output, first, False) as workflow:
        workflow.finish({"status": "PASS"})
    saved_manifest = (output / "RUN_MANIFEST.json").read_bytes()
    session = AnalysisSession(root, model, embeddings)
    versions["pydantic"] = "changed-parser"
    with pytest.raises(ValueError, match="environment changed"):
        session.validate_unchanged()
    second = analysis_fingerprint(root, model, embeddings, {})
    with (
        pytest.raises(ValueError, match="fingerprint mismatch"),
        CheckpointRun(output, second, True),
    ):
        pytest.fail("Stale workflow was reused")
    assert (output / "RUN_MANIFEST.json").read_bytes() == saved_manifest
    session.close(validate=False)


def test_training_fingerprint_tracks_parser_dependencies_without_training(versions):
    first = training_fingerprint({"manifest_sha256": "fixture"}, {}, FIXED_CONFIG, "cpu")
    versions["interegular"] = "changed-parser"
    second = training_fingerprint({"manifest_sha256": "fixture"}, {}, FIXED_CONFIG, "cpu")
    assert first["runtime_versions"]["interegular"] != second["runtime_versions"]["interegular"]
    assert first["config"] == second["config"]
