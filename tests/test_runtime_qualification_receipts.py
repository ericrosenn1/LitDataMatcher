"""Qualification failure receipts, using explicit runtime doubles, not model evidence."""

import json
from dataclasses import asdict

import pytest

from scripts.v2 import runtime_qualify


class DeniedProbe:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def connect(self, *args):
        raise PermissionError("test network denial")


class FakeRuntime:
    claims = []
    changed_identity = False
    changed_backend = False

    def __init__(self, model, config):
        self.config = config
        self.calls = 0

    def inference_fingerprint(self, document):
        return {"identity": "synthetic-expected", "config": asdict(self.config)}

    def backend_identity(self):
        return {
            "backend": "changed"
            if self.changed_backend and self.calls
            else "synthetic-test-double",
            "config": asdict(self.config),
        }

    def extract(self, document, cache, force_fresh=False):
        self.calls += 1
        fingerprint = self.inference_fingerprint(document)
        if self.changed_identity:
            fingerprint = {"identity": "synthetic-changed"}
        return {
            "claims": self.claims,
            "questions": [],
            "inference_manifest": {
                "origin": "fresh_local_inference" if force_fresh else "cache_replay",
                "fingerprint": fingerprint,
                "configuration": asdict(self.config),
                "elapsed_seconds": 0,
            },
        }


@pytest.fixture
def qualifier(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_qualify.sys, "addaudithook", lambda hook: None)
    monkeypatch.setattr(runtime_qualify.socket, "socket", lambda: DeniedProbe())
    monkeypatch.setattr(runtime_qualify, "LocalSemanticRuntime", FakeRuntime)
    monkeypatch.setattr(FakeRuntime, "claims", [])
    monkeypatch.setattr(FakeRuntime, "changed_identity", False)
    monkeypatch.setattr(FakeRuntime, "changed_backend", False)
    source = tmp_path / "development.json"
    source.write_text(
        json.dumps(
            {"document_id": "synthetic-development", "title": "Test", "text": "Test source."}
        )
    )
    output = tmp_path / "qualification.json"
    args = [
        "--model-dir",
        str(tmp_path),
        "--embedding-dir",
        str(tmp_path),
        "--document",
        str(source),
        "--output",
        str(output),
    ]
    return args, output


def test_no_accepted_records_writes_diagnostic_fail_receipt(qualifier):
    args, output = qualifier
    assert runtime_qualify.main(args) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "FAIL"
    assert report["fresh"]["claims"] == []
    assert report["failure"]["stage"] == "cached_replay"
    assert report["expected_identity"]["config"]["max_new_tokens"] == 1024
    assert report["replay_identity"] == report["expected_identity"]
    assert report["backend_identity"]["backend"] == "synthetic-test-double"
    assert report["replay_configuration"] == report["configuration"]


def test_changed_runtime_identity_cannot_qualify(qualifier, monkeypatch):
    args, output = qualifier
    monkeypatch.setattr(FakeRuntime, "claims", [{"id": "synthetic"}])
    monkeypatch.setattr(FakeRuntime, "changed_identity", True)
    assert runtime_qualify.main(args) == 1
    assert "identity differs" in json.loads(output.read_text())["failure"]["message"]


@pytest.mark.parametrize("mode", ["source_schema", "source_index", "source_literal"])
def test_source_profile_cli_records_explicit_configuration(qualifier, mode):
    args, output = qualifier
    assert runtime_qualify.main([*args, "--decoding-mode", mode, "--max-new-tokens", "384"]) == 1
    report = json.loads(output.read_text())
    assert report["configuration"]["decoding_mode"] == mode
    assert report["configuration"]["max_new_tokens"] == 384


def test_changed_static_backend_cannot_qualify(qualifier, monkeypatch):
    args, output = qualifier
    monkeypatch.setattr(FakeRuntime, "claims", [{"id": "synthetic"}])
    monkeypatch.setattr(FakeRuntime, "changed_backend", True)
    assert runtime_qualify.main(args) == 1
    assert "Static backend identity changed" in json.loads(output.read_text())["failure"]["message"]


def test_existing_receipt_is_preserved(qualifier):
    args, output = qualifier
    output.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        runtime_qualify.main(args)
    assert output.read_text() == "existing evidence"
