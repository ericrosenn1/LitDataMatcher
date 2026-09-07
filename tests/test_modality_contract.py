from litdatamatcher.modality_contract import compatibility, modality_contract


def test_contract_keeps_technical_runs_distinct_and_unknowns_explicit():
    row = {"assay_types": ["RNA-SEQ"], "organisms": ["Homo sapiens"], "access_type": "public metadata", "metadata": {"dependence": {"technical_run_count": 4, "donor_links": "AMBIGUOUS_NOT_INFERRED"}}}
    contract = modality_contract(row)
    assert contract["modality"] == ["bulk_transcriptomics"]
    assert contract["biological_unit"] == "UNKNOWN"
    assert compatibility("microbiome_metagenomics", "Homo sapiens", row) == "INCOMPATIBLE"
    assert compatibility("bulk_transcriptomics", "Mus musculus", row) == "INCOMPATIBLE"


def test_absent_metadata_is_unknown_not_incompatible():
    assert compatibility("bulk_transcriptomics", "Homo sapiens", {"assay_types": []}) == "UNKNOWN"
