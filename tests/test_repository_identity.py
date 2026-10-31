"""Source-identity fixtures do not execute models or evaluate scientific claims."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from litdatamatcher import provenance, semantic_runtime, v2
from litdatamatcher.phase2_benchmark import source_identity


@pytest.fixture
def repository(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("Actual repository fixture requires Git")
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-m",
            "Synthetic repository",
        ],
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    return root, commit


def test_actual_root_and_phase2_alias(repository):
    root, commit = repository
    assert source_identity is provenance.repository_identity
    assert provenance.repository_identity(root)["commit"] == commit
    assert v2._native_source_commit(root) == commit


def test_nested_export_cannot_claim_parent_commit(repository):
    root, _ = repository
    nested = root / "unpacked source"
    nested.mkdir()
    identity = provenance.repository_identity(nested)
    assert identity["commit"] is None and identity["git_error"] == "repository_root_mismatch"
    assert v2._native_source_commit(nested) == "installed-distribution"


@pytest.mark.parametrize(
    "error", [FileNotFoundError("Missing Git"), subprocess.CalledProcessError(128, ["git"])]
)
def test_unavailable_git_retains_native_sentinel(tmp_path, monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(provenance.subprocess, "check_output", unavailable)
    assert provenance.repository_identity(tmp_path)["commit"] is None
    assert v2._native_source_commit(tmp_path) == "installed-distribution"


@pytest.mark.parametrize("kind", ["root", "nested", "missing_git"])
def test_native_manifest_uses_root_checked_identity(repository, tmp_path, monkeypatch, kind):
    root, commit = repository
    if kind == "nested":
        root = root / "export"
        root.mkdir()
    if kind == "missing_git":

        def unavailable(*args, **kwargs):
            raise FileNotFoundError("Synthetic missing Git")

        monkeypatch.setattr(provenance.subprocess, "check_output", unavailable)
    resolve = provenance.repository_identity
    monkeypatch.setattr(v2, "repository_identity", lambda ignored: resolve(root))

    class Runtime:
        model_manifest = {"model_id": "fixture", "revision": "fixture", "license": "fixture"}

        def __init__(self, *args, **kwargs):
            pass

    class Index:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(semantic_runtime, "LocalSemanticRuntime", Runtime)
    monkeypatch.setattr(semantic_runtime, "PretrainedSemanticIndex", Index)
    data, out = tmp_path / "data", tmp_path / "run"
    for name in ("literature", "studies", "processed_inspections"):
        v2.write_rows(data / f"catalog/{name}.jsonl", [])
    result = v2.analyze(data, out, tmp_path / "model", tmp_path / "encoder")
    assert result["status"] == "PASS"
    manifest = json.loads((out / "RUN_MANIFEST.json").read_text())
    assert manifest["source"]["commit"] == (commit if kind == "root" else "installed-distribution")
    assert set(manifest["source"]) == {
        "repository",
        "commit",
        "working_tree_digest",
        "spec_digest",
        "config_digest",
    }
    schema = json.loads(
        (Path(v2.__file__).parent / "schemas_v2/RUN_MANIFEST.schema.json").read_text()
    )
    import jsonschema

    jsonschema.validate(manifest["source"], schema["properties"]["source"])
