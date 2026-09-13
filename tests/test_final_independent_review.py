"""Independent adversarial review fixtures; no source acquisition or model inference."""

import hashlib

import pytest

from litdatamatcher.adapters import search_dataset_sources, search_literature_sources
from litdatamatcher.semantic_runtime import _file_sha
from litdatamatcher.v2 import document_lifecycle_status, normalize_dataset


@pytest.fixture(autouse=True)
def deny_external_requests(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Independent review attempted networking")

    for target in ("requests.sessions.Session.request", "socket.create_connection", "socket.socket.connect"):
        monkeypatch.setattr(target, forbidden)


def test_declared_python310_support_does_not_require_python311_hash_helper(tmp_path, monkeypatch):
    path = tmp_path / "local-model-fixture.bin"
    path.write_bytes(b"local synthetic integrity fixture")
    # Python 3.10 lacks this helper; emulate that precise standard-library boundary.
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    assert _file_sha(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_biological_unit_metadata_remains_unknown_after_normalization():
    record = {"dataset_id": "synthetic-metadata-only", "title": "Synthetic metadata fixture",
              "assay_types": ["RNA-seq"], "organisms": ["Homo sapiens"],
              "source_provenance": {"source_url": "fixture:metadata"}}
    normalized = normalize_dataset(record)
    assert normalized["modality_contract"]["biological_unit"] == "UNKNOWN"


class PubMedFixture:
    def __init__(self, signal=None, efetch_error=False):
        self.signal = signal
        self.efetch_error = efetch_error

    def get_json(self, url, params=None):
        if "esearch" in url:
            return {"esearchresult": {"idlist": ["90000001"]}}
        row = {"title": "Synthetic source fixture", "articleids": []}
        if self.signal == "summary_type":
            row["pubtype"] = ["Retracted Publication"]
        return {"result": {"90000001": row}}

    def get_text(self, url, params=None):
        if self.efetch_error:
            raise FileNotFoundError("synthetic EFetch offline cache miss")
        article_type = '<PublicationTypeList><PublicationType UI="D016441">Retracted Publication</PublicationType></PublicationTypeList>' if self.signal == "xml_type" else ""
        relation = '<CommentsCorrectionsList><CommentsCorrections RefType="RetractionIn"><PMID>90000002</PMID></CommentsCorrections></CommentsCorrectionsList>' if self.signal == "xml_relation" else ""
        return f'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>90000001</PMID><Article><ArticleTitle>Synthetic source fixture</ArticleTitle><Abstract><AbstractText>Fixture metadata only.</AbstractText></Abstract>{article_type}</Article>{relation}</MedlineCitation></PubmedArticle></PubmedArticleSet>'


@pytest.mark.parametrize("signal", ["summary_type", "xml_type", "xml_relation"])
def test_pubmed_explicit_retraction_metadata_blocks_downstream_evidence(signal):
    rows = search_literature_sources("synthetic fixture", ["pubmed"], client=PubMedFixture(signal))
    assert len(rows) == 1
    assert document_lifecycle_status(rows[0]) == "RETRACTED"


def test_pubmed_failed_efetch_is_visible_even_when_summary_rows_are_retained():
    rows = search_literature_sources("synthetic fixture", ["pubmed"], client=PubMedFixture(efetch_error=True))
    assert len(rows) == 1
    assert rows.source_statuses[0]["status"] != "OBSERVED"


@pytest.mark.parametrize("source", ["openalex", "pubmed", "crossref", "mgnify", "ena"])
def test_provider_error_objects_never_become_successful_empty_searches(source):
    class Client:
        def get_json(self, url, params=None):
            return {"error": "synthetic provider failure"}

    search = search_dataset_sources if source in {"mgnify", "ena"} else search_literature_sources
    rows = search("synthetic fixture", [source], client=Client())
    assert len(rows) == 0
    assert rows.source_statuses[0]["status"] == "UNKNOWN_RETRIEVAL_OR_SCHEMA_FAILURE"
