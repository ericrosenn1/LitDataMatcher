"""Small synthetic tables validate inspection, not scientific findings."""

import pytest

from scripts.campaign_20261031.inspect_uci_data import download, inspect_table


def metadata():
    return {"num_instances": 2, "num_features": 1, "missing_values_symbol": "?",
            "target_col": ["label"], "index_col": None, "variables": [
                {"name": "age", "role": "Feature", "type": "Integer", "units": "years"},
                {"name": "label", "role": "Target", "type": "Categorical"},
            ]}


def test_missingness_and_nonfinite_remain_distinct_and_raw_is_untouched(tmp_path):
    path = tmp_path / "source.csv"
    content = "age,label\n?,A\ninf,B\n"
    path.write_text(content)
    report = inspect_table(path, metadata())
    assert report["rows"] == report["columns"] == 2
    assert report["missing_cells"] == report["nonfinite_cells"] == 1
    assert report["variables"]["age"]["units"] == "years"
    assert path.read_text() == content


@pytest.mark.parametrize("content", ["<html>blocked</html>", "age,label\n2,A\n", "age,label\nwrong,A\n4,B\n", "age,label\n3,A,extra\n4,B\n"])
def test_inaccessible_or_malformed_tables_fail(tmp_path, content):
    path = tmp_path / "source.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        inspect_table(path, metadata())


def test_unsupported_download_is_rejected_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: pytest.fail("Network must not be used"))
    with pytest.raises(ValueError, match="Unsupported"):
        download("https://example.invalid/reads.fastq", tmp_path / "x", set(), 100)
