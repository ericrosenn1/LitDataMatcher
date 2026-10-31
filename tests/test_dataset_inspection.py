"""Synthetic fixtures test evidence boundaries; live demonstrations are separate."""

import hashlib
import json

import pytest

from litdatamatcher import dataset_inspection as inspection
from litdatamatcher.biostudies_adapter import ArrayExpressDatasetAdapter, BioStudiesDatasetAdapter
from litdatamatcher.scientific_v2 import assess_requirements

SERIES = """^SERIES = GSE1
!Series_geo_accession = GSE1
!Series_title = Engineering fixture
!Series_type = Expression profiling by array
!Series_sample_id = GSM1
!Series_sample_id = GSM2
"""


def sample(identity, *, species="Homo sapiens", group="case"):
    return f"""^SAMPLE = {identity}
!Sample_geo_accession = {identity}
!Sample_series_id = GSE1
!Sample_organism_ch1 = {species}
!Sample_characteristics_ch1 = condition: {group}
!Sample_characteristics_ch1 = tissue: blood
!Sample_contact_email = example@example.com
"""


class Snapshot:
    payloads = {"GSE1": SERIES, "GSM1": sample("GSM1"), "GSM2": sample("GSM2", group="control")}

    def __init__(self, *args, **kwargs):
        self.events = []

    def get(self, url, params):
        raw = self.payloads[params["acc"]].encode()
        return raw, {
            "url": url + "?acc=" + params["acc"],
            "sha256": hashlib.sha256(raw).hexdigest(),
        }


def test_complete_sample_profile_does_not_claim_measurements_or_donors(tmp_path, monkeypatch):
    monkeypatch.setattr(inspection, "SnapshotClient", Snapshot)
    result = inspection.inspect_dataset({"dataset_id": "GSE1"}, tmp_path)
    assert result["inspection_status"] == "INSPECTED_METADATA"
    assert result["independent_units"] is None
    assert result["capabilities"]["outcome"]["status"] == "unknown"
    assert result["capabilities"]["sample_attribute:condition"]["value"] == ["case", "control"]
    assert "contact_email" not in result["samples"][0]["fields"]
    assert not result["inspection"]["processed_measurements_inspected"]
    score = assess_requirements(
        [
            {"field": "species", "expected": "human"},
            {"field": "assay", "expected": "bulk_transcriptomics"},
        ],
        result,
    )
    assert score["eligibility"] == "DIRECT_FIT"
    assert (
        assess_requirements([{"field": "outcome", "expected": "survival"}], result)["eligibility"]
        == "REQUIRES_INSPECTION"
    )


def test_prefix_coverage_does_not_generalize_to_whole_study(tmp_path, monkeypatch):
    monkeypatch.setattr(inspection, "SnapshotClient", Snapshot)
    result = inspection.inspect_dataset({"dataset_id": "GSE1"}, tmp_path, max_samples=1)
    assert result["inspection_status"] == "PARTIAL_METADATA"
    assert result["capabilities"]["species"]["status"] == "unknown"
    assert result["capabilities"]["tissue"]["status"] == "unknown"
    assert result["capabilities"]["sample_attribute:condition"]["value"] == ["case"]


def test_bad_sample_membership_preserves_partial_failure(tmp_path, monkeypatch):
    class Bad(Snapshot):
        payloads = dict(
            Snapshot.payloads, GSM2=sample("GSM2").replace("series_id = GSE1", "series_id = GSE2")
        )

    monkeypatch.setattr(inspection, "SnapshotClient", Bad)
    result = inspection.inspect_dataset({"dataset_id": "GSE1"}, tmp_path)
    assert result["inspection_status"] == "PARTIAL_METADATA"
    assert result["inspection"]["failures"][0]["id"] == "GSM2"


def test_offline_miss_is_failure_not_absence(tmp_path):
    result = inspection.inspect_dataset({"dataset_id": "GSE1"}, tmp_path, offline=True)
    assert result["inspection_status"] == "SOURCE_FAILURE"
    assert "capabilities" not in result


@pytest.mark.parametrize("identity", ["GSE279879"])
def test_reserved_studies_are_not_requested(tmp_path, identity):
    with pytest.raises(ValueError, match="Reserved"):
        inspection.inspect_dataset({"dataset_id": identity}, tmp_path)


def test_explicit_reserved_family_link_stops_samples(tmp_path, monkeypatch):
    class Reserved(Snapshot):
        payloads = {"GSE1": SERIES + "!Series_relation = SuperSeries of: GSE279879\n"}

    monkeypatch.setattr(inspection, "SnapshotClient", Reserved)
    result = inspection.inspect_dataset({"dataset_id": "GSE1"}, tmp_path)
    assert result["inspection_status"] == "SOURCE_FAILURE"
    assert result["inspection"]["samples_inspected"] == 0


def test_cached_data_respects_same_byte_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(inspection, "SnapshotClient", Snapshot)
    result = inspection.inspect_dataset(
        {"dataset_id": "GSE1"}, tmp_path, max_bytes=10, offline=True
    )
    assert result["inspection_status"] == "SOURCE_FAILURE"


@pytest.mark.parametrize(
    "key,value", [("max_samples", True), ("max_samples", 251), ("max_bytes", 0)]
)
def test_invalid_budgets_fail_before_io(tmp_path, key, value):
    with pytest.raises(ValueError):
        inspection.inspect_dataset({"dataset_id": "GSE1"}, tmp_path, **{key: value})


def metadata():
    return {
        "dataset_id": "fixture",
        "name": "Fixture table",
        "num_instances": 3,
        "variables": [
            {"name": "age", "role": "Feature", "type": "Integer", "units": "years"},
            {"name": "diagnosis", "role": "Target", "type": "Categorical"},
        ],
    }


def test_processed_table_observation_missingness_and_joint_rows(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("age,diagnosis\n30,A\n?,B\n50,A\n")
    result = inspection.inspect_tabular_dataset(
        path,
        metadata(),
        source_url="https://example.org/data.csv",
        metadata_url="https://example.org/dict",
    )
    assert result["inspection"]["missing_cells"] == 1
    assert result["inspection"]["joint_observed_rows"] == {"age|diagnosis": 2}
    assert result["inspection"]["variables"]["age"]["source_definition"]["units"] == "years"
    score = assess_requirements(
        [
            {"field": "measured_variable", "expected": "age"},
            {"field": "observed_group:diagnosis", "expected": "A"},
        ],
        result,
    )
    assert score["eligibility"] == "DIRECT_FIT"
    assert result["independent_units"] is None
    assert (
        assess_requirements([{"field": "measured_variable", "expected": "survival"}], result)[
            "eligibility"
        ]
        == "NOT_QUALIFIED"
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "text",
    [
        "age,diagnosis\n1,A\n",
        "age,diagnosis\n1,A,x\n2,B\n3,A\n",
        "age,diagnosis\ninvalid,A\n2,B\n3,A\n",
    ],
)
def test_table_drift_rejected(tmp_path, text):
    path = tmp_path / "table.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        inspection.inspect_tabular_dataset(
            path,
            metadata(),
            source_url="https://example.org/data.csv",
            metadata_url="https://example.org/dict",
        )


class Client:
    last_response_metadata = {}

    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def get_json(self, url, params=None):
        self.calls.append((url, params))
        return self.payload


def test_biostudies_keeps_files_and_search_hits_unqualified():
    client = Client(
        {
            "totalHits": 100,
            "hits": [
                {"accession": "E-MTAB-1", "title": "<b>Study</b>", "isPublic": True, "files": 4}
            ],
        }
    )
    adapter = ArrayExpressDatasetAdapter(client)
    result = adapter.search("diabetes")[0]
    assert client.calls[0][1]["collection"] == "arrayexpress"
    assert result.metadata["reported_file_count"] == 4
    assert result.sample_size == 0 and result.title == "Study"
    assert result.metadata["measurement_availability"] == "UNKNOWN_UNINSPECTED"
    assert adapter.last_search_status["reported_total_hits"] == 100


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"totalHits": 1, "hits": [{"accession": "../../secret", "title": "Study"}]},
        {"totalHits": 0, "hits": None},
    ],
)
def test_biostudies_schema_drift_is_not_empty_success(payload):
    with pytest.raises(ValueError):
        BioStudiesDatasetAdapter(Client(payload)).search("diabetes")


def test_matrix_identity_and_complete_table_required(tmp_path):
    text = (
        '!Series_geo_accession\t"GSE1"\n!Sample_geo_accession\t"GSM1"\n'
        '!series_matrix_table_begin\n"ID_REF"\t"GSM1"\n"probe1"\t1\n'
        "!series_matrix_table_end\n"
    )
    path = tmp_path / "matrix.txt"
    path.write_text(text)
    value = inspection.inspect_geo_matrix(path, "GSE1", source_url="https://example.org/matrix")
    assert value["processed_measurements_present"]
    for bad in (text.replace("!series_matrix_table_end", ""), text.replace('"GSE1"', '"GSE2"')):
        path.write_text(bad)
        with pytest.raises(ValueError):
            inspection.inspect_geo_matrix(path, "GSE1", source_url="https://example.org/matrix")


def test_series_inspection_replaces_stale_modality_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(inspection, "SnapshotClient", Snapshot)
    result = inspection.inspect_dataset(
        {"dataset_id": "GSE1", "modality_contract": {"modality": ["UNKNOWN"]}}, tmp_path
    )
    assert result["modality_contract"]["modality"] == ["bulk_transcriptomics"]
