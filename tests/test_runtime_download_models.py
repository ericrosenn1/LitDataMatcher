"""Model-file checksums remain byte-identical without Python 3.11 file_digest."""

import hashlib

import pytest

from scripts.v2.runtime_download_models import file_sha256


@pytest.mark.parametrize("size", [0, 3, 1024 * 1024 + 17])
def test_streaming_model_digest_matches_exact_bytes_on_python310(tmp_path, monkeypatch, size):
    content = (b"\x00\xffmodel-bytes\n" * (size // 14 + 1))[:size]
    path = tmp_path / "model.bin"
    path.write_bytes(content)
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    assert file_sha256(path) == hashlib.sha256(content).hexdigest()
