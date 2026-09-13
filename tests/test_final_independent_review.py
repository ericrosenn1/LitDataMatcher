"""Independent adversarial review fixtures; no source acquisition or model inference."""

import hashlib
import json
from copy import deepcopy

import pytest

from litdatamatcher.adapters import search_dataset_sources, search_literature_sources
from litdatamatcher.calibration_readiness import fit_validate_calibration, verify_calibration_result
from litdatamatcher.data_plane import digest
from litdatamatcher.literature_integrity import consolidate_literature_rows, invalidate_affected_derivations
from litdatamatcher.scientific_v2 import assess_requirements
from litdatamatcher.semantic_runtime import _file_sha
from litdatamatcher.v2 import analyze, document_lifecycle_status, normalize_dataset, read_rows, write_rows


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
        relation = '<CommentsCorrectionsList><CommentsCorrections RefType="RetractionIn"><RefSource>Synthetic fixture notice</RefSource><PMID>90000002</PMID></CommentsCorrections></CommentsCorrectionsList>' if self.signal == "xml_relation" else ""
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


@pytest.mark.parametrize("source", ["openalex", "crossref"])
def test_other_explicit_provider_retraction_signals_are_not_discarded(source):
    class Client:
        def get_json(self, url, params=None):
            if source == "openalex":
                return {"results": [{"id": "https://openalex.org/W90000001", "title": "Synthetic fixture", "is_retracted": True}]}
            return {"message": {"items": [{"DOI": "10.1234/fixture-notice", "title": ["Synthetic fixture notice"], "update-to": [{"DOI": "10.1234/fixture-original", "type": "retraction", "source": "publisher"}]}]}}

    rows = search_literature_sources("synthetic fixture", [source], client=Client())
    assert len(rows) == 1
    assert document_lifecycle_status(rows[0]) != "ACTIVE_METADATA_ONLY"


@pytest.mark.parametrize("changes,dependence,expected", [
    ({}, {}, "OBSERVED"),
    ({"mapping_type": "synonym"}, {}, "OBSERVED"),
    ({"mapping_type": "related"}, {}, "UNKNOWN"),
    ({"status": "derived"}, {}, "UNKNOWN"),
    ({"source_locator": None}, {}, "UNKNOWN"),
    ({"value": None, "status": "unknown"}, {}, "UNKNOWN"),
    ({}, {"donor_links": "AMBIGUOUS_NOT_INFERRED"}, "UNKNOWN"),
])
def test_biological_unit_requires_qualified_observation(changes, dependence, expected):
    observation = {"value": "participant", "status": "observed", "source_locator": "fixture:sample-table#unit", **changes}
    raw = {"dataset_id": "fixture-unit", "capabilities": {"biological_sample": observation}, "metadata": {"dependence": dependence}}
    assert normalize_dataset(raw)["modality_contract"]["biological_unit"] == expected


@pytest.mark.parametrize("observation,expected", [
    ({"value": None, "status": "absent", "source_locator": "fixture:protocol#sample-not-collected"}, "NOT_QUALIFIED"),
    ({"value": "participant", "status": "known", "source_locator": ["fixture:sample-table#unit"]}, "DIRECT_FIT"),
])
def test_biological_unit_repair_preserves_explicit_absence_and_legacy_migration(observation, expected):
    raw = {"dataset_id": "fixture-declared-unit", "capabilities": {"biological_sample": observation}}
    result = assess_requirements([{"field": "biological_sample", "expected": "participant"}], normalize_dataset(raw))
    assert result["eligibility"] == expected


@pytest.mark.parametrize("observed,expected,status", [
    ("NCBITaxon:9606", "human", "DIRECT_FIT"),
    ("NCBITaxon:10090", "human", "NOT_QUALIFIED"),
    ("Homo sapiens (Human)", "human", "DIRECT_FIT"),
    ("Mus musculus (Mouse)", "human", "NOT_QUALIFIED"),
    ("Drosophila melanogaster (Fruit fly)", "Drosophila melanogaster", "DIRECT_FIT"),
    ("synthetic unresolved organism", "human", "REQUIRES_INSPECTION"),
    ("synthetic unresolved organism", "synthetic unresolved organism", "REQUIRES_INSPECTION"),
    ("Homo sapiens", "synthetic unresolved organism", "REQUIRES_INSPECTION"),
])
def test_taxonomy_repair_retains_unknown_and_qualified_identifier_semantics(observed, expected, status):
    raw = {"dataset_id": "fixture-taxon", "organisms": [observed], "source_provenance": {"source_url": "fixture:organism"}}
    result = assess_requirements([{"field": "species", "expected": expected}], normalize_dataset(raw))
    assert result["eligibility"] == status


@pytest.mark.parametrize("observed,expected", [("RNA-seq", "EFO:0002772"), ("EFO:0002772", "RNA-seq")])
def test_qualified_assay_identifier_survives_precise_assay_guard(observed, expected):
    raw = {"dataset_id": "fixture-assay-identifier", "assay_types": [observed], "source_provenance": {"source_url": "fixture:assay"}}
    result = assess_requirements([{"field": "assay", "expected": expected}], normalize_dataset(raw))
    assert result["eligibility"] == "DIRECT_FIT"


def literature_pages(second_hash="b", cache_status="LIVE"):
    class Pages:
        last_response_metadata = {}

        def get_json(self, url, params):
            second = params["cursorMark"] != "*"
            self.last_response_metadata = {"cache_content_sha256": (second_hash if second else "a") * 64, "cache_status": cache_status}
            item = {"id": "2" if second else "1", "source": "MED", "title": "Synthetic fixture", "abstractText": "Synthetic source metadata."}
            return {"resultList": {"result": [item] if second else [{"id": "skipped-invalid-row"}, item]}, **({} if second else {"nextCursorMark": "second"})}

    return search_literature_sources("synthetic fixture", ["europepmc"], client=Pages(), limit=101)


def test_page_provenance_survives_skipped_rows_and_invalidates_only_changed_page():
    before = literature_pages()
    after = literature_pages("c")
    assert [r["source_provenance"]["metadata"]["cache_snapshot"]["cache_content_sha256"] for r in before] == ["a" * 64, "b" * 64]
    states = [invalidate_affected_derivations(old, new, ["fixture-claim"])["status"] for old, new in zip(before, after, strict=True)]
    assert states == ["UNCHANGED", "INVALIDATED"]
    replay = literature_pages(cache_status="HIT")
    assert all(invalidate_affected_derivations(old, new, ["fixture-claim"])["status"] == "UNCHANGED" for old, new in zip(before, replay, strict=True))


@pytest.mark.parametrize("source,payload", [
    ("openalex", {"results": []}),
    ("pubmed", {"esearchresult": {"idlist": []}}),
    ("crossref", {"message": {"items": []}}),
    ("mgnify", {"data": []}),
    ("ena", []),
    ("europepmc", {"resultList": {"result": []}}),
    ("clinicaltrials", {"studies": []}),
])
def test_valid_empty_envelopes_remain_successful_searches(source, payload):
    class Client:
        def get_json(self, url, params=None):
            return deepcopy(payload)

    search = search_dataset_sources if source in {"mgnify", "ena", "clinicaltrials"} else search_literature_sources
    rows = search("synthetic fixture", [source], client=Client())
    assert rows == []
    assert rows.source_statuses[0]["status"] == "OBSERVED"


def test_pubmed_ordinary_comment_does_not_inherit_retraction_from_citation_text():
    class CommentFixture(PubMedFixture):
        def get_text(self, url, params=None):
            return super().get_text(url, params).replace("RetractionIn", "CommentOn").replace("Synthetic fixture notice", "Retraction discussed in another publication")

    rows = search_literature_sources("synthetic fixture", ["pubmed"], client=CommentFixture("xml_relation"))
    assert document_lifecycle_status(rows[0]) == "ACTIVE_METADATA_ONLY"


def calibration_inputs():
    """Independent numerical fixture: no real labels or expert result asserted."""
    rows = []
    for role in ("FIT", "VALIDATION"):
        for group, label in (("a", 1), ("b", 0)):
            for index, score in enumerate((0.8, 0.9)):
                rows.append({
                    "record_id": f"{role}-{group}-{index}", "group_id": f"{role}-{group}",
                    "split_role": role, "split_family": "independent-review-fixture", "record_status": "RETAINED",
                    "label_origin": "source_determined", "data_origin": "synthetic_fixture",
                    "label_provenance": {"source_locator": "fixture:label", "label_definition": "synthetic binary outcome"},
                    "split_provenance": {"source_locator": "fixture:partition"}, "dimension": "relevance",
                    "label": label, "score": score, "score_definition": "synthetic score", "ablation": "full",
                })
    protocol = {"protocol_id": "independent-review-fixture-v1", "source_locator": "fixture:protocol",
                "split_family": "independent-review-fixture", "dimension": "relevance", "label_definition": "synthetic binary outcome",
                "score_definition": "synthetic score", "method": "isotonic_pava_step_v1", "predeclared": True,
                "data_origin": "synthetic_fixture", "minimum_records_per_split": 4, "minimum_groups_per_split": 2,
                "maximum_brier_score": 0.3, "minimum_brier_improvement": 0.05}
    update_partition(rows, protocol)
    return rows, protocol


def update_partition(rows, protocol):
    protocol["partition_digest"] = digest([{key: row[key] for key in ("record_id", "group_id", "split_role")} for row in sorted(rows, key=lambda row: row["record_id"])])


def test_group_replication_cannot_inflate_isotonic_fit_or_validation_weight():
    rows, protocol = calibration_inputs()
    baseline = fit_validate_calibration(rows, protocol=protocol)
    for row in list(rows):
        if row["group_id"].endswith("-a"):
            rows.extend([{**row, "record_id": f"{row['record_id']}-replicate-{n}"} for n in range(3)])
    update_partition(rows, protocol)
    replicated = fit_validate_calibration(rows, protocol=protocol)
    assert baseline["model"] == replicated["model"]
    assert replicated["validation"]["calibrated_brier"] == baseline["validation"]["calibrated_brier"] == 0.25
    assert replicated["validation"]["raw_brier"] == pytest.approx(baseline["validation"]["raw_brier"])
    assert replicated["calibration_status"] == "SYNTHETIC_VALIDATION_ONLY"
    assert not verify_calibration_result(replicated, require_real_expert=True)


def test_self_consistent_hash_tampering_still_requires_recomputed_calibration():
    rows, protocol = calibration_inputs()
    result = fit_validate_calibration(rows, protocol=protocol)
    assert verify_calibration_result(result)
    forged = deepcopy(result)
    forged["model"]["blocks"][0]["probability"] = 0.9
    forged["model_digest"] = digest(forged["model"])
    assert not verify_calibration_result(forged)


def analyze_fixture(tmp_path, monkeypatch, fail_extraction=False):
    import litdatamatcher.semantic_runtime as runtime_module

    processed_views = []

    class Runtime:
        model_manifest = {"model_id": "independent-review-fixture", "revision": "fixture", "license": "fixture"}

        def __init__(self, *args, **kwargs):
            pass

        def extract(self, view, *args, **kwargs):
            processed_views.append(view)
            if fail_extraction:
                raise RuntimeError("Injected synthetic extraction failure")
            claim = {"claim_id": "fixture-local-claim", "subject": "treatment", "predicate": "changed", "object": "response",
                     "statement": view["text"], "status": "direct_experiment", "direction": "increase", "context": "fixture",
                     "evidence_span": {"start": 0, "end": len(view["text"]), "text": view["text"]}}
            return {"claims": [claim], "questions": [], "rejected": [],
                    "inference_manifest": {"origin": "synthetic_test_stub", "fingerprint": {"prompt_sha256": "fixture"}}}

    class Index:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, records):
            self.records = records

        def search(self, *args, **kwargs):
            return [{"id": row["id"], "score": 1.0 if "wrong" in row["id"] else -1.0} for row in self.records]

    monkeypatch.setattr(runtime_module, "LocalSemanticRuntime", Runtime)
    monkeypatch.setattr(runtime_module, "PretrainedSemanticIndex", Index)
    text = "Treatment changed clinical response."
    prefix = "Synthetic introduction. "
    active = {"document_id": "fixture-active", "title": "Synthetic source", "text": prefix + text,
              "source_locator": "fixture:active", "split_context": "development",
              "sections": [{"start": len(prefix), "end": len(prefix + text), "text": text, "section": "Results"}]}
    blocked = {**active, "document_id": "fixture-retracted", "version_relationships": {"is-retracted-by": [{"id": "fixture-notice"}]}}
    reserved = {**active, "document_id": "fixture-reserved", "split_context": "synthetic-reserved-family"}
    studies = [{"dataset_id": identifier, "title": "Clinical treatment response", "organisms": [species],
                "source_provenance": {"source_url": f"fixture:{identifier}"}, "split_context": "development"}
               for identifier, species in (("fixture-compatible", "Homo sapiens"), ("fixture-wrong-species", "Mus musculus"))]
    root = tmp_path / "synthetic-root"
    out = tmp_path / "synthetic-run"
    write_rows(root / "catalog/literature.jsonl", [active, blocked, reserved])
    write_rows(root / "catalog/studies.jsonl", studies)
    write_rows(root / "catalog/processed_inspections.jsonl", [])
    result = analyze(root, out, tmp_path / "unused-model", tmp_path / "unused-embedding",
                     question="Did treatment change clinical response?", requirements=[{"field": "species", "expected": "human"}],
                     limit=3, chunks=1, device="cpu")
    return result, out, processed_views, active


def test_analyze_composes_lifecycle_qualification_spans_and_dossier_with_runtime_stub(tmp_path, monkeypatch):
    result, out, processed_views, active = analyze_fixture(tmp_path, monkeypatch)
    assert len(processed_views) == 1
    assert read_rows(out / "excluded_documents.jsonl") == [{"document_id": "fixture-retracted", "reason": "RETRACTED"}]
    claim = read_rows(out / "claims.jsonl")[0]
    span = claim["evidence_span"]
    assert active["text"][span["start"]:span["end"]] == span["text"]
    assert claim["source_document_id"] == active["document_id"]
    matches = read_rows(out / "matches.jsonl")
    assert [row["dataset_id"] for row in matches] == ["fixture-compatible", "fixture-wrong-species"]
    assert matches[0]["assessment"]["eligibility"] == "DIRECT_FIT"
    assert matches[1]["assessment"]["eligibility"] == "NOT_QUALIFIED"
    dossiers = read_rows(out / "scientific_dossiers.jsonl")
    assert len(dossiers) == 2
    assert all(row["review_status"] == "SOURCE_ASSISTED_PENDING_EXPERT_REVIEW" for row in dossiers)
    assert all(row["unresolvedness"]["gap_status"] == "insufficient-coverage" for row in dossiers)
    assert all(row["source_evidence"][0]["integration_mode"] == "CONTEXT_ONLY_OR_UNRESOLVED" for row in dossiers)
    assert result["status"] == "PASS"  # Only fixture execution; no model or science validation.
    manifest = json.loads((out / "RUN_MANIFEST.json").read_text())
    assert manifest["inference"]["fresh_calls"] == 0
    assert manifest["inference"]["backend_qualified"] is False
    assert all(item["validation"] == "PASS" for item in manifest["artifacts"])
    assert all(hashlib.sha256((out / item["path"]).read_bytes()).hexdigest() == item["sha256"] for item in manifest["artifacts"])


def test_analyze_persists_partial_receipt_for_failed_extraction_without_phantom_dossiers(tmp_path, monkeypatch):
    result, out, processed_views, _ = analyze_fixture(tmp_path, monkeypatch, fail_extraction=True)
    manifest = json.loads((out / "RUN_MANIFEST.json").read_text())
    assert len(processed_views) == 1
    assert result["status"] == manifest["execution_status"] == "PARTIAL"
    assert manifest["failures"][0]["stage"] == "inference"
    assert not read_rows(out / "claims.jsonl")
    assert not read_rows(out / "scientific_dossiers.jsonl")
