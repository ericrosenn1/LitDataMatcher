import hashlib

import pytest

from litdatamatcher.data_plane import Catalog, atomic_json


@pytest.mark.parametrize("failures,winerror", [(2, 5), (6, 32), (1, None)])
def test_atomic_publication_handles_bounded_windows_sharing(
    tmp_path, monkeypatch, failures, winerror
):
    from litdatamatcher import data_plane

    path = tmp_path / "progress.json"
    path.write_bytes(b"old")
    replace, attempts = data_plane.os.replace, []

    def publish(source, target):
        attempts.append(1)
        assert path.read_bytes() == b"old"
        if len(attempts) <= failures:
            exc = PermissionError("denied")
            if winerror is not None:
                exc.winerror = winerror
            raise exc
        return replace(source, target)

    monkeypatch.setattr(data_plane.os, "replace", publish)
    monkeypatch.setattr(data_plane.time, "sleep", lambda _: None)
    if failures == 2:
        data_plane.atomic_write(path, b"new")
        assert path.read_bytes() == b"new"
        assert len(attempts) == 3
    else:
        with pytest.raises(PermissionError):
            data_plane.atomic_write(path, b"new")
        assert path.read_bytes() == b"old"
        assert len(attempts) == (6 if winerror else 1)
    assert list(tmp_path.iterdir()) == [path]


def test_invalidation_preserves_unrelated_and_replays(tmp_path):
    c = Catalog(tmp_path)
    assert c.upsert("document", "d", {"text": "first"})
    c.upsert("claim", "a", {"claim": "a"}, [("document", "d")])
    c.upsert("match", "m", {"match": "m"}, [("claim", "a")])
    c.upsert("dataset", "s", {"title": "untouched"}, search_text="human inflammatory response")
    assert not c.upsert("document", "d", {"text": "first"})
    assert len(c.records("match")) == 1
    c.upsert("document", "d", {"text": "corrected"})
    assert not c.records("claim") and not c.records("match")
    assert len(c.records("dataset")) == 1
    assert c.search("dataset", "inflammatory") == ["s"]
    with pytest.raises(ValueError):
        c.upsert("match", "x", {}, [("claim", "a")])
    c.close()


def test_snapshot_detects_corruption_and_finite_json(tmp_path):
    c = Catalog(tmp_path)
    row = c.snapshot(b"source", {"url": "https://example.org"})
    assert row["sha256"] == hashlib.sha256(b"source").hexdigest()
    from pathlib import Path

    Path(row["path"]).write_bytes(b"broken")
    with pytest.raises(ValueError):
        c.snapshot(b"source", {})
    with pytest.raises(ValueError):
        atomic_json(tmp_path / "bad.json", {"score": float("nan")})
    assert not (tmp_path / "bad.json").exists()
    c.close()
