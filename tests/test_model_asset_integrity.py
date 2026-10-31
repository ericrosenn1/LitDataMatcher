"""Synthetic file contracts only; no pretrained weights or model generation."""

import hashlib
import json
from pathlib import Path

import pytest

from litdatamatcher import semantic_runtime as runtime
from litdatamatcher.data_plane import digest as content_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def model(root, *, sharded=False):
    root.mkdir(exist_ok=True)
    files = {"config.json": b"{}", "tokenizer.json": b"{}"}
    if sharded:
        files.update({"part-1.safetensors": b"fixture1", "part-2.safetensors": b"fixture2"})
        files["model.safetensors.index.json"] = json.dumps(
            {"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}
        ).encode()
    else:
        files["model.safetensors"] = b"synthetic fixture, not loadable weights"
    for name, value in files.items():
        (root / name).write_bytes(value)
    manifest = {
        "model_id": "local/fixture",
        "revision": "a" * 40,
        "license": "synthetic",
        "files": [
            {"path": name, "bytes": len(value), "sha256": sha(root / name)}
            for name, value in files.items()
        ],
    }
    write(root / "MODEL_MANIFEST.json", manifest)
    return manifest


def replace_manifest(root, manifest):
    write(root / "MODEL_MANIFEST.json", manifest)


def update_index(root, manifest, value):
    write(root / "model.safetensors.index.json", value)
    for row in manifest["files"]:
        if row["path"] == "model.safetensors.index.json":
            row.update(sha256=sha(root / row["path"]), bytes=(root / row["path"]).stat().st_size)
    replace_manifest(root, manifest)


def training_sidecars(root, manifest):
    fingerprint = "b" * 64
    digest = content_digest(
        {"files": manifest["files"], "training_fingerprint": fingerprint, "epoch": 2}
    )
    manifest.update(
        revision_type="local_training_content_fingerprint",
        training_fingerprint=fingerprint,
        training_content_sha256=digest,
        revision=digest[:40],
    )
    replace_manifest(root, manifest)
    (root / "optimizer.pt").write_bytes(b"training-only fixture")
    receipt = {
        "epoch": 2,
        "training_fingerprint": fingerprint,
        "model_manifest_sha256": sha(root / "MODEL_MANIFEST.json"),
        "optimizer_sha256": sha(root / "optimizer.pt"),
    }
    write(root / "EPOCH_RECEIPT.json", receipt)
    return receipt


@pytest.mark.parametrize("sharded", [False, True])
def test_complete_manifest_returns_unchanged_and_hashes_each_asset_once(
    tmp_path, monkeypatch, sharded
):
    manifest = model(tmp_path, sharded=sharded)
    before = (tmp_path / "MODEL_MANIFEST.json").read_bytes()
    calls = []
    original = runtime._file_sha
    monkeypatch.setattr(runtime, "_file_sha", lambda path: (calls.append(path), original(path))[1])
    assert runtime.verify_model(tmp_path) == manifest
    assert len(calls) == len(set(calls)) == len(manifest["files"])
    assert (tmp_path / "MODEL_MANIFEST.json").read_bytes() == before


@pytest.mark.parametrize(
    "name",
    [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "vocab.txt",
        "generation_config.json",
        "nested/arbitrary.bin",
        "adapter_config.json",
    ],
)
def test_unlisted_inference_assets_are_rejected(tmp_path, name):
    manifest = model(tmp_path)
    manifest["files"] = [row for row in manifest["files"] if row["path"] != name]
    if name == "model.safetensors":
        (tmp_path / "dummy.safetensors").write_bytes(b"declared dummy")
        manifest["files"].append(
            {"path": "dummy.safetensors", "sha256": sha(tmp_path / "dummy.safetensors")}
        )
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b"unlisted asset")
    replace_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="Unlisted model asset"):
        runtime.verify_model(tmp_path)


def test_single_checkpoint_cannot_shadow_fully_listed_shards(tmp_path):
    model(tmp_path, sharded=True)
    (tmp_path / "model.safetensors").write_bytes(b"unlisted preferred checkpoint")
    with pytest.raises(ValueError, match="Unlisted model asset: model.safetensors"):
        runtime.verify_model(tmp_path)


@pytest.mark.parametrize(
    "reference",
    [
        "../outside.safetensors",
        "/outside.safetensors",
        "C:/outside.safetensors",
        "part-1.safetensors/../part-2.safetensors",
        "config.json",
    ],
)
def test_index_references_are_bounded_listed_safetensors(tmp_path, reference):
    manifest = model(tmp_path, sharded=True)
    update_index(tmp_path, manifest, {"weight_map": {"a": reference}})
    with pytest.raises((ValueError, FileNotFoundError)):
        runtime.verify_model(tmp_path)


@pytest.mark.parametrize("mapping", [{}, [], {"a": 2}])
def test_malformed_index_is_rejected(tmp_path, mapping):
    manifest = model(tmp_path, sharded=True)
    update_index(tmp_path, manifest, {"weight_map": mapping})
    with pytest.raises(ValueError):
        runtime.verify_model(tmp_path)


def test_index_must_not_reference_unlisted_documentation(tmp_path):
    manifest = model(tmp_path, sharded=True)
    (tmp_path / "README.md").write_text("not weights")
    update_index(tmp_path, manifest, {"weight_map": {"a": "README.md"}})
    with pytest.raises(ValueError, match="unlisted shard"):
        runtime.verify_model(tmp_path)


def test_documentation_and_known_download_metadata_are_allowed(tmp_path):
    manifest = model(tmp_path)
    for name in (
        "README.md",
        "LICENSE",
        ".cache/huggingface/.gitignore",
        ".cache/huggingface/download/config.json.metadata",
        ".cache/huggingface/download/model.safetensors.lock",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"bookkeeping")
    assert runtime.verify_model(tmp_path) == manifest
    (tmp_path / ".cache/huggingface/download/model.safetensors").write_bytes(b"not metadata")
    with pytest.raises(ValueError, match="Unlisted"):
        runtime.verify_model(tmp_path)


def test_training_sidecars_are_bound_without_optimizer_rehash(tmp_path, monkeypatch):
    manifest = model(tmp_path)
    training_sidecars(tmp_path, manifest)
    original = runtime._file_sha
    calls = []

    def checked(path):
        calls.append(Path(path).name)
        assert Path(path).name != "optimizer.pt"
        return original(path)

    monkeypatch.setattr(runtime, "_file_sha", checked)
    assert runtime.verify_model(tmp_path) == manifest
    assert len(calls) == len(manifest["files"])
    # Inference does not certify training optimizer bytes; validate_epoch does.
    (tmp_path / "optimizer.pt").write_bytes(b"changed training-only file")
    assert runtime.verify_model(tmp_path) == manifest
    (tmp_path / "model.shadow.safetensors").write_bytes(b"not a training sidecar")
    with pytest.raises(ValueError, match="Unlisted"):
        runtime.verify_model(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_manifest_sha256", "0" * 64),
        ("optimizer_sha256", "invalid"),
        ("epoch", True),
        ("training_fingerprint", "c" * 64),
    ],
)
def test_unbound_training_receipts_are_rejected(tmp_path, field, value):
    manifest = model(tmp_path)
    receipt = training_sidecars(tmp_path, manifest)
    receipt[field] = value
    write(tmp_path / "EPOCH_RECEIPT.json", receipt)
    with pytest.raises(ValueError, match="sidecars"):
        runtime.verify_model(tmp_path)


def test_training_fingerprint_cannot_be_claimed_without_correct_content(tmp_path):
    manifest = model(tmp_path)
    receipt = training_sidecars(tmp_path, manifest)
    manifest["training_content_sha256"] = "f" * 64
    replace_manifest(tmp_path, manifest)
    receipt["model_manifest_sha256"] = sha(tmp_path / "MODEL_MANIFEST.json")
    write(tmp_path / "EPOCH_RECEIPT.json", receipt)
    with pytest.raises(ValueError, match="training content"):
        runtime.verify_model(tmp_path)


def test_optimizer_without_qualified_receipt_is_rejected(tmp_path):
    model(tmp_path)
    (tmp_path / "optimizer.pt").write_bytes(b"unbound")
    with pytest.raises((ValueError, FileNotFoundError)):
        runtime.verify_model(tmp_path)


def test_listed_tamper_and_duplicate_paths_still_fail(tmp_path):
    manifest = model(tmp_path)
    (tmp_path / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="integrity"):
        runtime.verify_model(tmp_path)
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest["files"].append(dict(manifest["files"][0]))
    replace_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="Duplicate"):
        runtime.verify_model(tmp_path)


def test_symlink_shard_escape_is_rejected(tmp_path):
    root = tmp_path / "model"
    model(root, sharded=True)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"fixture1")
    path = root / "part-1.safetensors"
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("OS does not permit fixture symlinks")
    with pytest.raises(ValueError, match="inside the model root"):
        runtime.verify_model(root)


@pytest.mark.parametrize(
    "filename,key,value",
    [
        ("tokenizer_config.json", "tokenizer_file", "../external-tokenizer.json"),
        ("tokenizer_config.json", "vocab_file", "C:/external-vocab.json"),
        ("tokenizer_config.json", "fast_tokenizer_files", ["../tokenizer.4.0.json"]),
        ("config.json", "configuration_files", ["../config.4.0.json"]),
    ],
)
def test_configuration_references_cannot_escape_manifest(tmp_path, filename, key, value):
    manifest = model(tmp_path)
    if filename == "tokenizer_config.json":
        manifest["files"].append({"path": filename})
    write(tmp_path / filename, {key: value})
    for row in manifest["files"]:
        if row["path"] == filename:
            row.update(sha256=sha(tmp_path / filename), bytes=(tmp_path / filename).stat().st_size)
    replace_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="asset path"):
        runtime.verify_model(tmp_path)


def test_configuration_reference_to_listed_asset_is_accepted(tmp_path):
    manifest = model(tmp_path)
    filename = "tokenizer_config.json"
    write(tmp_path / filename, {"tokenizer_file": "tokenizer.json"})
    manifest["files"].append({"path": filename, "sha256": sha(tmp_path / filename)})
    replace_manifest(tmp_path, manifest)
    assert runtime.verify_model(tmp_path) == manifest


def test_training_exception_does_not_authorize_inference_file_references(tmp_path):
    manifest = model(tmp_path)
    path = tmp_path / "tokenizer_config.json"
    write(path, {"tokenizer_file": "optimizer.pt"})
    manifest["files"].append({"path": path.name, "sha256": sha(path)})
    training_sidecars(tmp_path, manifest)
    with pytest.raises(ValueError, match="unlisted model asset"):
        runtime.verify_model(tmp_path)


def test_listed_adapter_configuration_cannot_redirect_to_external_base(tmp_path):
    manifest = model(tmp_path)
    path = tmp_path / "adapter_config.json"
    write(path, {"base_model_name_or_path": "../outside", "peft_type": "LORA"})
    manifest["files"].append({"path": path.name, "sha256": sha(path)})
    replace_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="Adapter configuration"):
        runtime.verify_model(tmp_path)
