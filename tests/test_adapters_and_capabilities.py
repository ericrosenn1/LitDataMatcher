import json

import pytest

from litdatamatcher.adapters import (
    ClinicalTrialsDatasetAdapter,
    CrossrefLiteratureAdapter,
    ENASRADatasetAdapter,
    EuropePMCLiteratureAdapter,
    GEODatasetAdapter,
    MGnifyDatasetAdapter,
    PubMedLiteratureAdapter,
    search_dataset_sources,
    search_literature_sources,
)
from litdatamatcher.capability_registry import (
    CAPABILITY_VOCABULARY,
    capability_summary,
    infer_dataset_capabilities,
)
from litdatamatcher.cli import main
from litdatamatcher.datasets import CuratedBiomedicalCatalogAdapter, classify_dataset_record
from litdatamatcher.http_cache import CachedHttpClient
from litdatamatcher.provenance import summarize_source_provenance
from litdatamatcher.storage import read_jsonl, write_jsonl


class FakeClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get_json(self, url, params=None, **kwargs):
        self.calls.append((url, params or {}))
        return self.payloads.pop(0)

    def get_text(self, url, params=None, **kwargs):
        self.calls.append((url, params or {}))
        return self.payloads.pop(0)


def test_pubmed_literature_adapter_normalizes_esummary_rows():
    client = FakeClient(
        [
            {"esearchresult": {"idlist": ["123"]}},
            {
                "result": {
                    "123": {
                        "title": "Microbiome recovery after antibiotics",
                        "pubdate": "2025 Jan",
                        "articleids": [{"idtype": "doi", "value": "10.1/example"}],
                    }
                }
            },
            """<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>123</PMID>
      <Article>
        <Journal><Title>Example Journal</Title><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Microbiome recovery after antibiotics</ArticleTitle>
        <Abstract><AbstractText>Future studies should examine longitudinal recovery.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Smith</LastName><Initials>J</Initials></Author></AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/xml</ArticleId></ArticleIdList></PubmedData>
  </PubmedArticle>
</PubmedArticleSet>""",
        ]
    )

    rows = PubMedLiteratureAdapter(client).search_literature("microbiome", limit=5)

    assert rows[0]["source_id"] == "pubmed:123"
    assert rows[0]["abstract"] == "Future studies should examine longitudinal recovery."
    assert rows[0]["doi"] == "10.1/xml"
    assert rows[0]["year"] == 2025
    assert rows[0]["metadata"]["authors"] == ["Smith J"]


def test_europepmc_adapter_normalizes_identifiers_versions_and_malformed_rows():
    client = FakeClient(
        [{"resultList": {"result": [
            {"source": "MED", "id": "123", "pmcid": "PMC123", "doi": "10.1/Example", "title": "Versioned study", "abstractText": "Abstract", "firstPublicationDate": "2025-01-02", "commentCorrectionList": {"commentCorrection": [{"id": "456"}]}},
            {"source": "MED", "id": "", "title": "Malformed missing id"},
        ]}}]
    )

    rows = EuropePMCLiteratureAdapter(client).search_literature("study", limit=5)

    assert len(rows) == 1
    assert rows[0]["source_id"] == "europepmc:MED:123"
    assert rows[0]["doi"] == "10.1/example"
    assert rows[0]["year"] == 2025
    assert rows[0]["version_relationships"]["commentCorrection"][0]["id"] == "456"
    assert rows[0]["source_provenance"]["retrieval_time_utc"]


def test_crossref_adapter_skips_missing_doi_and_preserves_update_metadata():
    client = FakeClient(
        [{"message": {"items": [
            {"DOI": "https://doi.org/10.2/Example", "title": ["Corrected study"], "published-online": {"date-parts": [[2024, 2, 1]]}, "relation": {"is-correction-of": [{"id": "10.2/original"}]}, "indexed": {"date-time": "2025-01-01T00:00:00Z"}, "author": [{"given": "Ada", "family": "Lovelace"}]},
            {"title": ["Missing DOI"]},
        ]}}]
    )

    rows = CrossrefLiteratureAdapter(client).search_literature("study", limit=5)

    assert len(rows) == 1
    assert rows[0]["source_id"] == "crossref:10.2/example"
    assert rows[0]["year"] == 2024
    assert rows[0]["metadata"]["authors"] == ["Ada Lovelace"]
    assert rows[0]["version_relationships"]["is-correction-of"][0]["id"] == "10.2/original"


def test_literature_search_merges_cross_source_doi_and_version_relations():
    client = FakeClient(
        [
            {"resultList": {"result": [{"source": "MED", "id": "123", "doi": "10.3/shared", "title": "Shared"}]}},
            {"message": {"items": [{"DOI": "10.3/shared", "title": ["Shared"], "relation": {"is-version-of": [{"id": "10.3/old"}]}}]}},
        ]
    )

    rows = search_literature_sources("shared", ["europepmc", "crossref"], client=client, limit=5)

    assert len(rows) == 1
    assert rows[0]["metadata"]["alternate_source_ids"] == ["crossref:10.3/shared"]
    assert rows[0]["metadata"]["version_relationships"]["crossref"]["is-version-of"][0]["id"] == "10.3/old"


def test_cached_http_client_offline_replays_and_fails_closed_on_miss(tmp_path, monkeypatch):
    client = CachedHttpClient(cache_dir=tmp_path)
    url = "https://example.test/works"
    params = {"query": "one"}
    client._cache_path(url, params).write_text('{"message": "cached"}', encoding="utf-8")
    offline = CachedHttpClient(cache_dir=tmp_path, offline=True)
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: pytest.fail("network attempted"))

    assert offline.get_json(url, params=params) == {"message": "cached"}
    with pytest.raises(FileNotFoundError, match="offline cache missing"):
        offline.get_json(url, params={"query": "missing"})


def test_geo_dataset_adapter_normalizes_dataset_records():
    client = FakeClient(
        [
            {"esearchresult": {"idlist": ["200"]}},
            {
                "result": {
                    "200": {
                        "accession": "GSE200",
                        "title": "IBD transcriptomics",
                        "summary": "RNA-seq study of treatment response.",
                        "gdstype": "Expression profiling by high throughput sequencing",
                        "n_samples": "42",
                    }
                }
            },
        ]
    )

    records = GEODatasetAdapter(client).search("IBD transcriptomics")

    assert records[0].dataset_id == "GSE200"
    assert records[0].source == "GEO"
    assert records[0].sample_size == 42
    assert any(variable.name == "transcriptomics" for variable in records[0].variables)


def test_clinicaltrials_adapter_preserves_registry_design_without_sample_claims():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT01234567", "briefTitle": "Active treatment study"},
            "statusModule": {"overallStatus": "RECRUITING", "lastUpdatePostDateStruct": {"date": "2026-01-02"}},
            "descriptionModule": {"briefSummary": "Registry summary."},
            "conditionsModule": {"conditions": ["Inflammatory bowel disease"]},
            "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE2"], "enrollmentInfo": {"count": 120, "type": "ESTIMATED"}, "designInfo": {"allocation": "RANDOMIZED", "interventionModel": "PARALLEL"}},
            "armsInterventionsModule": {"armGroups": [{"label": "Placebo", "type": "PLACEBO_COMPARATOR", "interventionNames": ["Placebo"]}], "interventions": [{"name": "Drug A", "type": "DRUG"}]},
            "outcomesModule": {"primaryOutcomes": [{"measure": "Clinical remission", "timeFrame": "Week 12"}]},
            "eligibilityModule": {"sex": "ALL", "minimumAge": "18 Years", "maximumAge": "65 Years", "healthyVolunteers": False, "eligibilityCriteria": "Eligible adults."},
        }
    }
    record = ClinicalTrialsDatasetAdapter(FakeClient([{"studies": [study]}])).search("ibd")[0]

    assert record.dataset_id == "NCT01234567"
    assert record.sample_size == 0
    assert record.metadata["enrollment"]["count"] == 120
    assert record.metadata["enrollment"]["interpretation"] == "registry enrollment; not analyzed sample count"
    assert record.metadata["study_type"] == "INTERVENTIONAL"
    assert record.metadata["primary_outcomes"][0]["time_frame"] == "Week 12"
    assert record.metadata["comparators"][0]["label"] == "Placebo"
    assert record.metadata["eligibility_population"]["sex"] == "ALL"
    assert any(variable.name == "timepoint" for variable in record.variables)


def test_clinicaltrials_observational_and_duplicate_versions_remain_nonperturbational():
    def study(version, status):
        return {"protocolSection": {"identificationModule": {"nctId": "NCT76543210", "briefTitle": "Observational cohort"}, "statusModule": {"overallStatus": status, "lastUpdatePostDateStruct": {"date": version}}, "designModule": {"studyType": "OBSERVATIONAL", "enrollmentInfo": {"count": "20"}}, "conditionsModule": {}, "armsInterventionsModule": {}, "outcomesModule": {}, "eligibilityModule": {}}}

    record = ClinicalTrialsDatasetAdapter(FakeClient([{"studies": [study("2024-01-01", "COMPLETED"), study("2025-01-01", "RECRUITING"), {"protocolSection": {"identificationModule": {"nctId": "bad"}}}]}])).search("cohort")[0]

    assert record.metadata["causal_interpretation"] == "NOT_PERTURBATIONAL"
    assert record.metadata["overall_status"] == "RECRUITING"
    assert record.metadata["alternate_versions"] == ["2024-01-01"]
    assert record.metadata["missingness"]["outcomes"] == "MISSING"


def test_dataset_search_cli_replays_clinicaltrials_cache_offline(tmp_path, capsys, monkeypatch):
    cache = CachedHttpClient(cache_dir=tmp_path / "cache")
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {"query.term": "study", "pageSize": 25, "format": "json"}
    cache._cache_path(url, params).write_text(json.dumps({"studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT11111111", "briefTitle": "Cached study"}, "designModule": {"studyType": "OBSERVATIONAL"}}}]}), encoding="utf-8")
    out = tmp_path / "records.jsonl"
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: pytest.fail("network attempted"))

    assert main(["dataset-search", "--query", "study", "--source", "clinicaltrials", "--cache-dir", str(cache.cache_dir), "--offline", "--out", str(out)]) == 0
    assert read_jsonl(out)[0]["metadata"]["causal_interpretation"] == "NOT_PERTURBATIONAL"
    assert json.loads(capsys.readouterr().out)["records"] == 1


def test_ena_adapter_groups_runs_without_equating_them_to_samples():
    rows = [
        {"study_accession": "ERP12345", "secondary_study_accession": "PRJEB123", "study_title": "Sequencing study", "run_accession": "ERR1", "experiment_accession": "ERX1", "sample_accession": "ERS1", "secondary_sample_accession": "SAME1", "sample_title": "Sample one", "scientific_name": "Homo sapiens", "library_strategy": "RNA-SEQ", "fastq_ftp": "ftp://x", "last_updated": "2025-01-01"},
        {"study_accession": "ERP12345", "run_accession": "ERR2", "experiment_accession": "ERX2", "sample_accession": "ERS1", "scientific_name": "Homo sapiens", "library_strategy": "RNA-SEQ", "last_updated": "2025-01-02"},
        {"study_accession": "bad", "run_accession": "ERR_BAD"},
        "schema-drift-row",
    ]
    record = ENASRADatasetAdapter(FakeClient([rows])).search("ignored")[0]

    assert record.dataset_id == "ERP12345"
    assert record.sample_size == 0
    assert record.metadata["dependence"]["technical_run_count"] == 2
    assert record.metadata["dependence"]["biological_sample_count"] == 1
    assert record.metadata["dependence"]["donor_links"] == "AMBIGUOUS_NOT_INFERRED"
    assert record.metadata["access_availability"]["raw_reads"] is True
    assert record.metadata["pagination"]["status"] == "BOUNDED_PAGE_NOT_COMPLETE_CENSUS"


def test_ena_adapter_deduplicates_repeated_run_links_and_missing_fields():
    rows = [{"study_accession": "SRP12345", "study_title": "Run duplicate", "run_accession": "SRR1", "sample_accession": "SRS1"}, {"study_accession": "SRP12345", "study_title": "Run duplicate", "run_accession": "SRR1", "sample_accession": "SRS1"}]
    record = ENASRADatasetAdapter(FakeClient([{"data": rows}])).search("ignored")[0]

    assert record.metadata["runs"] == ["SRR1"]
    assert len(record.metadata["run_sample_links"]) == 1
    assert record.metadata["missingness"]["organism"] == "MISSING"


def test_dataset_search_cli_replays_ena_cache_offline(tmp_path, capsys, monkeypatch):
    cache = CachedHttpClient(cache_dir=tmp_path / "cache")
    url = "https://www.ebi.ac.uk/ena/portal/api/search"
    params = {"result": "read_run", "query": "study", "fields": "study_accession,secondary_study_accession,secondary_project,study_title,study_alias,run_accession,experiment_accession,sample_accession,secondary_sample_accession,sample_alias,sample_title,sample_description,scientific_name,library_strategy,library_source,library_selection,fastq_ftp,submitted_ftp,sra_ftp,first_public,last_updated", "format": "json", "limit": 100}
    cache._cache_path(url, params).write_text(json.dumps([{"study_accession": "ERP77777", "study_title": "Cached ENA", "run_accession": "ERR7", "sample_accession": "ERS7"}]), encoding="utf-8")
    out = tmp_path / "ena.jsonl"
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: pytest.fail("network attempted"))

    assert main(["dataset-search", "--query", "study", "--source", "ena", "--cache-dir", str(cache.cache_dir), "--offline", "--out", str(out)]) == 0
    assert read_jsonl(out)[0]["dataset_id"] == "ERP77777"
    assert json.loads(capsys.readouterr().out)["records"] == 1


def test_mgnify_dataset_adapter_normalizes_json_api_rows():
    client = FakeClient(
        [
            {
                "count": 1,
                "items": [
                    {
                        "accession": "MGYS0001",
                        "study_name": "Gut microbiome study",
                        "abstract": "Metagenomic profiles from human gut samples.",
                        "sample_count": 12,
                        "experiment_type": "metagenomics",
                    }
                ]
            }
        ]
    )

    records = MGnifyDatasetAdapter(client).search("gut microbiome")

    assert records[0].dataset_id == "MGYS0001"
    assert records[0].source == "MGnify"
    assert records[0].sample_size == 0
    assert records[0].metadata["declared_sample_count"]["count"] == 12
    assert records[0].variables[0].name == "microbiome_composition"
    assert "api/v2/studies" in client.calls[0][0]


def test_search_helpers_deduplicate_and_limit_rows():
    literature_client = FakeClient(
        [
            {"esearchresult": {"idlist": ["1", "2"]}},
            {
                "result": {
                    "1": {"title": "One", "articleids": []},
                    "2": {"title": "Two", "articleids": []},
                }
            },
            "<PubmedArticleSet />",
        ]
    )
    dataset_client = FakeClient(
        [
            {"esearchresult": {"idlist": ["1"]}},
            {"result": {"1": {"accession": "GSE1", "title": "One", "summary": "RNA-seq"}}},
        ]
    )

    literature = search_literature_sources("query", ["pubmed"], client=literature_client, limit=1)
    datasets = search_dataset_sources("query", ["geo"], client=dataset_client, limit=1)

    assert len(literature) == 1
    assert len(datasets) == 1


def test_curated_catalog_records_carry_advisory_source_provenance():
    record = CuratedBiomedicalCatalogAdapter().records[0]
    provenance = record.metadata["source_provenance"]
    summary = summarize_source_provenance([record.to_dict()])

    assert provenance["source_type"] == "curated_biomedical_catalog"
    assert provenance["content_scope"] == "dataset_metadata"
    assert provenance["acquisition_method"] == "bundled_curated_catalog"
    assert provenance["status"] == "warning"
    assert summary["records_without_provenance"] == 0
    assert summary["source_types"]["curated_biomedical_catalog"] == 1


def test_curated_catalog_separates_capability_categories():
    records = {record.dataset_id: record for record in CuratedBiomedicalCatalogAdapter().records}
    qiita = records["qiita_microbiome_antibiotics_longitudinal"]
    geo = records["geo_ibd_transcriptomics"]

    qiita_categories = {variable.name: variable.category for variable in qiita.variables}
    geo_categories = {variable.name: variable.category for variable in geo.variables}

    assert qiita_categories["sample_size"] == "study_design_feature"
    assert qiita_categories["longitudinal_time"] == "temporal_structure"
    assert qiita_categories["predictor"] == "derived_or_proxy_capability"
    assert geo_categories["class_label"] == "supervised_learning_label"
    assert geo_categories["outcome_label"] == "evaluation_outcome"
    assert qiita.metadata["capability_caveats"]
    assert geo.metadata["capability_annotations"]


def test_capability_vocabulary_contains_ml_readiness_terms():
    assert CAPABILITY_VOCABULARY["sample_size"]["category"] == "study_design_feature"
    assert CAPABILITY_VOCABULARY["class_label"]["category"] == "supervised_learning_label"
    assert CAPABILITY_VOCABULARY["predictor"]["category"] == "derived_or_proxy_capability"
    assert CAPABILITY_VOCABULARY["prediction_performance"]["category"] == "evaluation_outcome"


def test_capability_registry_infers_observed_and_derived_capabilities():
    dataset = classify_dataset_record(
        {
            "dataset_id": "clinical-1",
            "title": "Treatment outcome cohort",
            "source": "Example",
            "variables": [
                {"name": "treatment", "category": "clinical", "completeness": 0.9},
                {"name": "outcome", "category": "clinical", "completeness": 0.8},
                {"name": "timepoint", "category": "design", "completeness": 0.7},
            ],
        }
    )

    capabilities = infer_dataset_capabilities(dataset)
    variables = {capability.variable_name for capability in capabilities}
    summary = capability_summary(capabilities)

    assert "treatment" in variables
    assert "treatment_response" in variables
    assert "longitudinal_change" in variables
    assert summary["vocabulary_terms"]
    assert summary["capabilities_by_type"]["derived"] >= 2


def test_capability_export_cli_writes_capability_jsonl(tmp_path, capsys):
    dataset = classify_dataset_record(
        {
            "dataset_id": "clinical-1",
            "title": "Treatment outcome cohort",
            "source": "Example",
            "variables": [
                {"name": "treatment", "category": "clinical", "completeness": 0.9},
                {"name": "outcome", "category": "clinical", "completeness": 0.8},
            ],
        }
    )
    datasets_path = tmp_path / "datasets.jsonl"
    out_path = tmp_path / "capabilities.jsonl"
    write_jsonl(datasets_path, [dataset.to_dict()])

    result = main(["capability-export", "--datasets", str(datasets_path), "--out", str(out_path)])
    captured = json.loads(capsys.readouterr().out)

    assert result == 0
    assert captured["capabilities"] >= 3
    assert any(row["capability_type"] == "derived" for row in read_jsonl(out_path))
