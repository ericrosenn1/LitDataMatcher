import gzip
import json

import pytest
import requests

from litdatamatcher.acquisition_v2 import SnapshotClient, parse_article, parse_series_matrix, profile_capabilities


class Response:
    def __init__(self, status=200, content=b"test", headers=None, interrupted=False):
        self.status_code, self.content = status, content
        self.headers = headers or {}
        self.interrupted = interrupted

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("injected", response=self)

    def iter_content(self, size):
        yield self.content
        if self.interrupted:
            raise requests.exceptions.ChunkedEncodingError("truncated")


class Session:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return next(self.responses)


def test_retry_after_transient_and_network_free_replay(tmp_path, monkeypatch):
    session = Session(Response(429, headers={"Retry-After": "1"}), Response(503), Response(content=b'{"ok":true}'))
    delays = []
    client = SnapshotClient(tmp_path, session=session, sleep=delays.append)
    data, meta = client.get("https://example.test/resource")
    assert data == b'{"ok":true}' and meta["attempts"] == 3
    assert delays == [1, 4]
    monkeypatch.setattr(requests, "Session", lambda: pytest.fail("offline attempted network session"))
    assert SnapshotClient(tmp_path, offline=True).get("https://example.test/resource")[0] == data
    assert len(list((tmp_path / "objects").iterdir())) == 1


def test_interrupted_download_never_installs_partial_object(tmp_path):
    session = Session(Response(content=b"partial", interrupted=True), Response(content=b"complete"))
    data, _ = SnapshotClient(tmp_path, session=session, sleep=lambda x: None).get("https://example.test/file")
    assert data == b"complete"
    assert [p.read_bytes() for p in (tmp_path / "objects").iterdir()] == [b"complete"]


def test_refresh_keeps_immutable_old_bytes(tmp_path):
    url = "https://example.test/v"
    SnapshotClient(tmp_path, session=Session(Response(content=b"v1"))).get(url)
    SnapshotClient(tmp_path, refresh=True, session=Session(Response(content=b"v2"))).get(url)
    assert {p.read_bytes() for p in (tmp_path / "objects").iterdir()} == {b"v1", b"v2"}


def test_cache_corruption_and_missing_fail_offline(tmp_path):
    client = SnapshotClient(tmp_path, session=Session(Response()))
    _, meta = client.get("https://example.test/a")
    (tmp_path / "objects" / meta["sha256"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt snapshot"):
        SnapshotClient(tmp_path, offline=True).get("https://example.test/a")
    with pytest.raises(FileNotFoundError, match="offline snapshot missing"):
        SnapshotClient(tmp_path, offline=True).get("https://example.test/b")


def test_limit_and_permanent_error_are_not_retried(tmp_path):
    session = Session(Response(content=b"too big"))
    with pytest.raises(ValueError, match="bound"):
        SnapshotClient(tmp_path, session=session, max_bytes=1).get("https://example.test/a")
    assert session.calls == 1
    session = Session(Response(404))
    with pytest.raises(requests.HTTPError):
        SnapshotClient(tmp_path, session=session).get("https://example.test/b")
    assert session.calls == 1
    assert not (tmp_path / "objects").exists()


MATRIX = '''!Sample_geo_accession\t"GSM1"\t"GSM2"
!Sample_title\t"donor99 LPS control"\t"donor98 treatment"
!Sample_characteristics_ch1\t"treatment: vehicle"\t"treatment: LPS"
!Sample_organism_ch1\t"Homo sapiens"\t"Homo sapiens"
!series_matrix_table_begin
"ID_REF"\t"GSM1"\t"GSM2"
"gene1"\t1\t2
"gene2"\tNA\t3
!series_matrix_table_end
'''


def test_matrix_actual_values_alignment_and_unknown_donors():
    samples, inspection = parse_series_matrix(gzip.compress(MATRIX.encode()), "GSE1")
    assert inspection["status"] == "PASS"
    assert inspection["sample_alignment"] and inspection["numeric_cells"] == 3
    assert inspection["missing_cells"] == 1
    assert all(s["donor_id"] is None for s in samples)
    assert [s["group"] for s in samples] == ["vehicle", "LPS"]
    record = {"source_locator": "GSE1", "organism": "Homo sapiens", "assay": "array"}
    profile_capabilities(record, samples)
    assert record["independent_units"] is None
    assert record["capabilities"]["intervention"]["value"] == ["LPS", "vehicle"]
    assert record["capabilities"]["comparator"]["status"] == "unknown"


def test_matrix_mismatch_nonfinite_and_metadata_only_are_not_pass():
    _, bad = parse_series_matrix(MATRIX.replace('"ID_REF"\t"GSM1"\t"GSM2"', '"ID_REF"\t"GSM2"\t"GSM1"').encode(), "GSE1")
    assert bad["status"] != "PASS"
    _, bad = parse_series_matrix(MATRIX.replace('"gene1"\t1\t2', '"gene1"\tinf\t2').encode(), "GSE1")
    assert bad["status"] != "PASS" and bad["invalid_cells"] == 1
    _, metadata = parse_series_matrix(MATRIX.split("!series_matrix_table_begin")[0].encode(), "GSE1")
    assert not metadata["processed_measurements_present"]
    assert metadata["status"] != "PASS"


def test_paragraph_spans_recover_qualifiers_without_duplication():
    paragraph = "Treatment did not increase expression. " * 10
    xml = f'<article><body><sec><title>Results</title><p>{paragraph}</p><sec><title>Null findings</title><p>Unknown donor linkage.</p></sec></sec></body></article>'.encode()
    result = parse_article(xml)
    assert len(result["sections"]) == 2
    for span in result["sections"]:
        assert result["text"][span["start"]:span["end"]] == span["text"]
    assert result["text"].count("Unknown donor linkage.") == 1
    with pytest.raises(ValueError, match="body missing"):
        parse_article(b"<article><front/></article>")
