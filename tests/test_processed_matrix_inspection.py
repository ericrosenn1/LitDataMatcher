"""Synthetic fixtures test parsing contracts; they are not biomedical evidence."""

import gzip

import pytest

from scripts.campaign_20261031.inspect_processed_data import inspect_matrix, validate_source_url

MATRIX = """!Series_title\t"Example fixture"
!Sample_geo_accession\t"GSM1"\t"GSM2"
!Sample_characteristics_ch1\t"group: case"\t"group: control"
!series_matrix_table_begin
"ID_REF"\t"GSM1"\t"GSM2"
"probe1"\t1\tNA
"probe2"\tNaN\t2
!series_matrix_table_end
"""


def test_matrix_dimensions_preserve_missing_and_nonfinite(tmp_path):
    path = tmp_path / "GSE1_series_matrix.txt.gz"
    path.write_bytes(gzip.compress(MATRIX.encode()))
    result = inspect_matrix(path)
    assert result["features"] == result["sample_columns"] == 2
    assert result["cells"] == 4
    assert result["finite_numeric_cells"] == 2
    assert result["missing_cells"] == result["nonfinite_cells"] == 1
    assert result["reported_characteristic_labels"] == ["group"]
    assert result["sample_metadata_fields"]["Sample_characteristics_ch1"][0] == [
        "group: case",
        "group: control",
    ]
    assert result["source_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace('"probe2"\tNaN\t2', '"probe2"\t2'),
        lambda s: s.replace('"probe2"', '"probe1"'),
        lambda s: s.replace("!series_matrix_table_end", ""),
        lambda s: s.replace('"probe2"\tNaN\t2', '"probe2"\tabc\t2'),
        lambda s: "<html>Access denied</html>",
    ],
)
def test_malformed_or_inaccessible_content_is_not_a_matrix(tmp_path, change):
    path = tmp_path / "GSE1_series_matrix.txt.gz"
    path.write_bytes(gzip.compress(change(MATRIX).encode()))
    with pytest.raises(ValueError):
        inspect_matrix(path)


def test_raw_sequencing_and_untrusted_urls_are_refused():
    with pytest.raises(ValueError):
        validate_source_url("https://example.com/GSE1_series_matrix.txt.gz", "GSE1")
    with pytest.raises(ValueError):
        validate_source_url("https://ftp.ncbi.nlm.nih.gov/geo/reads/reads.fastq.gz", "GSE1")


def test_declared_sample_ids_must_match_columns(tmp_path):
    path = tmp_path / "GSE1_series_matrix.txt"
    path.write_text(
        MATRIX.replace(
            '!Sample_geo_accession\t"GSM1"\t"GSM2"', '!Sample_geo_accession\t"GSM2"\t"GSM1"'
        )
    )
    with pytest.raises(ValueError, match="accessions disagree"):
        inspect_matrix(path)
