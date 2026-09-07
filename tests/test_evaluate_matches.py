import importlib.util
from pathlib import Path

MODULE = Path(__file__).parents[1] / "benchmarks" / "v2" / "evaluate_matches.py"
SPEC = importlib.util.spec_from_file_location("evaluate_matches", MODULE)
evaluate_matches = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluate_matches)


def row(dataset_id: str, lineage: list[str], topic: str = "primary") -> dict:
    return {"dataset_id": dataset_id, "topic": topic, "study_lineage": lineage, "organism": "Homo sapiens",
            "assay": "Expression profiling by high throughput sequencing", "title": "Source-described inflammatory response",
            "summary": "A source summary", "sample_count_reported": 7, "source_locator": "https://example.test/" + dataset_id,
            "source_snapshot": {"sha256": "a" * 64, "url": "https://example.test/snapshot"}}


def test_selection_excludes_reserved_and_linked_family_before_scoring() -> None:
    records = [row("GSE-reserved", ["GSE-reserved"]), row("GSE-a", ["GSE-a", "PRJNA-a"]),
               row("GSE-copy", ["GSE-copy", "PRJNA-a"]), row("GSE-b", ["GSE-b", "PRJNA-b"])]
    selected = evaluate_matches.selected_records(records, "primary", 2, {"GSE-reserved"})
    assert [item["dataset_id"] for item in selected] == ["GSE-a", "GSE-b"]


def test_profile_preserves_unknown_comparator_and_does_not_infer_donors() -> None:
    profile = evaluate_matches.source_profile(row("GSE-a", ["GSE-a"]))
    assert profile["independent_units"] is None
    assert profile["capabilities"]["comparator"]["status"] == "unknown"
    audit = evaluate_matches.capability_audit([profile])
    assert audit["comparators_retained_unknown"] == 1
    assert audit["independent_donor_counts_inferred"] == 0


def test_metrics_count_negative_candidates_and_invalid_top_match() -> None:
    result = evaluate_matches.ranking_metrics(["wrong", "right"], {"wrong": 0, "right": 3})
    assert result["recall_at_10_numerator"] == 1
    assert result["invalid_top_match"] == 1
