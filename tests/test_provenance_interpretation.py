from litdatamatcher.provenance import (
    adapter_source_profile_table,
    check_provenance_transfer,
    curated_catalog_provenance,
    module_boundary_map,
    module_ownership_registry,
    parser_caveat_records,
    parser_caveats,
    provenance_interpretation,
    provenance_review_caveats,
    source_profile,
    summarize_source_provenance,
)


def test_curated_catalog_provenance_is_advisory_and_deterministic():
    provenance = curated_catalog_provenance(
        dataset_id="dataset-1",
        source_name="Example Repository",
        source_url="https://example.org/dataset",
    ).to_dict()
    caveats = provenance_review_caveats([provenance])

    assert provenance["source_type"] == "curated_biomedical_catalog"
    assert provenance["content_scope"] == "dataset_metadata"
    assert provenance["acquisition_method"] == "bundled_curated_catalog"
    assert provenance["status"] == "warning"
    assert provenance["retrieval_time_utc"] == ""
    assert any("verified against source repositories" in caveat for caveat in caveats)


def test_parser_caveats_standardize_missing_body_and_fallback_warnings():
    caveats = parser_caveats(
        "generic_xml",
        body_text="",
        abstract="",
        sections=[],
        section_records=[],
        fallback=True,
    )

    assert caveats["parse_quality"] == "metadata_only_or_empty"
    assert "missing_body_text" in caveats["warning_codes"]
    assert "generic_xml_fallback" in caveats["warning_codes"]
    assert "no_offsets" in caveats["limitation_codes"]
    assert "no_figures_tables" in caveats["limitation_codes"]
    assert any(record["code"] == "no_offsets" for record in caveats["limitation_records"])


def test_parser_caveat_records_are_inspectable_by_code():
    records = parser_caveat_records()
    codes = {record["code"] for record in records}

    assert {"abstract_only", "missing_body_text", "generic_xml_fallback"} <= codes
    assert {"no_offsets", "no_figures_tables", "pdf_parse_partial"} <= codes


def test_source_profiles_and_reviewer_caveats_capture_metadata_only_limits():
    profile = source_profile("pubmed")
    caveats = provenance_review_caveats(
        [
            {
                "source_type": "pubmed",
                "content_scope": "abstract_plus_metadata",
                "warnings": ["No abstract was available or inferred."],
                "limitations": [],
            }
        ]
    )

    assert profile["category"] == "literature_metadata"
    assert profile["acquisition_method"] == "ncbi_eutilities"
    assert "pmid" in profile["native_id_fields"]
    assert any("abstract-level" in caveat.lower() for caveat in caveats)
    assert "No abstract was available or inferred." in caveats


def test_adapter_profiles_and_module_ownership_are_inspectable():
    profiles = adapter_source_profile_table()
    ownership = module_ownership_registry()

    assert set(profiles) == {"clinicaltrials", "crossref", "europepmc", "geo", "mgnify", "openalex", "pubmed"}
    assert profiles["clinicaltrials"]["native_id_fields"] == ["nct_id"]
    assert ownership["xml_parsing"]["owner_module"] == "litdatamatcher.literature_xml"
    assert ownership["matching"]["owner_module"] == "litdatamatcher.ranking"


def test_source_summary_and_transfer_check_preserve_handoff_semantics():
    provenance = {
        "source_type": "pubmed",
        "content_scope": "abstract_plus_metadata",
        "acquisition_method": "ncbi_eutilities",
        "status": "warning",
        "warnings": ["No abstract was available or inferred."],
        "limitations": [],
    }
    source_record = {"source_id": "source-1", "source_provenance": provenance}
    question = {
        "question_id": "question-1",
        "metadata": {"source_provenance": provenance},
    }
    dataset = {
        "dataset_id": "dataset-1",
        "metadata": {"source_provenance": provenance},
    }
    review_record = {
        "match_id": "match-1",
        "source_provenance": [provenance],
        "source_caveats": provenance_review_caveats([provenance]),
    }
    interpretation = provenance_interpretation([provenance])
    summary = summarize_source_provenance([question, dataset])
    check = check_provenance_transfer(
        source_records=[source_record],
        questions=[question],
        datasets=[dataset],
        review_records=[review_record],
        report_summary=summary,
    )

    assert summary["records_with_provenance"] == 2
    assert summary["review_caveats"]
    assert interpretation["profiles"][0]["source_type"] == "pubmed"
    assert interpretation["caveats"]
    assert check["status"] == "pass"
    assert check["stages"]["question_metadata"]["with_provenance"] == 1
    assert check["stages"]["review_visibility"]["source_caveat_records"] == 1
    assert check["module_ownership"]["review_export"]["owner_module"] == "litdatamatcher.review"
    assert "litdatamatcher.adapters" in module_boundary_map()
