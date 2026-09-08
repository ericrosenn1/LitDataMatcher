from litdatamatcher.modality_contract import compatibility, modality_contract
from litdatamatcher.ranking import rank_matches, score_question_dataset
from litdatamatcher.schemas import DatasetRecord, DatasetVariable, QuestionCandidate
from litdatamatcher.scientific_v2 import assess_requirements
from litdatamatcher.v2 import normalize_dataset


def test_contract_keeps_technical_runs_distinct_and_unknowns_explicit():
    row = {"assay_types": ["RNA-SEQ"], "organisms": ["Homo sapiens"], "access_type": "public metadata", "metadata": {"dependence": {"technical_run_count": 4, "donor_links": "AMBIGUOUS_NOT_INFERRED"}}}
    contract = modality_contract(row)
    assert contract["modality"] == ["bulk_transcriptomics"]
    assert contract["biological_unit"] == "UNKNOWN"
    assert compatibility("microbiome_metagenomics", "Homo sapiens", row) == "INCOMPATIBLE"
    assert compatibility("bulk_transcriptomics", "Mus musculus", row) == "INCOMPATIBLE"


def test_absent_metadata_is_unknown_not_incompatible():
    assert compatibility("bulk_transcriptomics", "Homo sapiens", {"assay_types": []}) == "UNKNOWN"


def test_scientific_contract_blocks_explicit_mismatch_and_technical_units():
    dataset = {
        "dataset_id": "ENA:study",
        "modality_contract": {
            "modality": ["sequencing_genomics"],
            "organisms": ["Homo sapiens"],
            "biological_unit": "UNKNOWN",
            "technical_units": 12,
        },
        "capabilities": {
            "modality": {"value": "sequencing_genomics", "status": "observed", "source_locator": "adapter"},
            "organism": {"value": "Homo sapiens", "status": "observed", "source_locator": "adapter"},
            "biological_sample_count": {"value": 12, "status": "observed", "source_locator": "technical runs"},
        },
    }
    mismatch = assess_requirements(
        [{"field": "modality", "expected": "bulk_transcriptomics"}], dataset
    )
    organism = assess_requirements(
        [{"field": "organism", "expected": "Mus musculus"}], dataset
    )
    units = assess_requirements(
        [{"field": "biological_sample_count", "expected": 12}], dataset
    )
    assert mismatch["eligibility"] == "NOT_QUALIFIED"
    assert organism["eligibility"] == "NOT_QUALIFIED"
    assert units["eligibility"] == "REQUIRES_INSPECTION"
    assert units["requirements"][0]["status"] == "UNKNOWN"


def test_v2_normalizer_attaches_contract_and_direct_profiles_stay_guarded():
    raw = {
        "dataset_id": "ENA:raw-study",
        "assay_types": ["WGS"],
        "organisms": ["Homo sapiens"],
        "metadata": {"dependence": {"donor_links": "AMBIGUOUS_NOT_INFERRED"}},
        "capabilities": {},
    }
    normalized = normalize_dataset(raw)
    assert normalized["modality_contract"]["modality"] == ["sequencing_genomics"]
    assert assess_requirements(
        [{"field": "modality", "expected": "bulk_transcriptomics"}], normalized
    )["eligibility"] == "NOT_QUALIFIED"
    assert assess_requirements(
        [{"field": "organism", "expected": "Mus musculus"}], raw
    )["eligibility"] == "NOT_QUALIFIED"


def test_ranking_excludes_explicit_contract_mismatch_but_keeps_unknown_reviewable():
    question = QuestionCandidate(
        question_id="q-contract",
        question="Does the assay support the question?",
        required_variables=["feature"],
        metadata={"required_modality": "bulk_transcriptomics", "required_organism": "Homo sapiens"},
    )
    incompatible = DatasetRecord(
        dataset_id="genome", title="Genome study", source="fixture",
        assay_types=["WGS"], organisms=["Homo sapiens"],
        variables=[DatasetVariable(name="feature")],
    )
    unknown = DatasetRecord(
        dataset_id="unknown", title="Unannotated study", source="fixture",
        variables=[DatasetVariable(name="feature")],
    )
    _, _, _, assessment = score_question_dataset(question, incompatible)
    assert assessment["modality_contract"]["status"] == "INCOMPATIBLE"
    matches = rank_matches([question], [incompatible, unknown])
    assert [match.dataset.dataset_id for match in matches] == ["unknown"]
