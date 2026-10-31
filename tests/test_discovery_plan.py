"""Controlled source-node failures validate acquisition, not scientific accuracy."""

import json

import pytest

from litdatamatcher.discovery import discover, validate_plan


def plan():
    return {
        "schema_version": "discovery-plan-1.0",
        "queries": [
            {
                "id": "q1",
                "query": "heart disease",
                "dataset_sources": ["geo", "datacite"],
                "limit": 3,
            }
        ],
    }


def result(job, cache, offline):
    return {
        "status": "SUCCESS",
        "records": [
            {
                "dataset_id": "study:1",
                "title": "A study",
                "source": job["source"],
                "description": "Measurement\u2028with Unicode",
                "organisms": ["human"],
                "metadata": {},
            }
        ],
    }


def test_independent_nodes_and_verified_resume(tmp_path):
    calls = []

    def fetch(job, cache, offline):
        calls.append((job["job_id"], cache))
        return result(job, cache, offline)

    first = discover(plan(), tmp_path, workers=2, _fetcher=fetch)
    assert first["status"] == "PASS"
    assert len(calls) == 2 and calls[0][1] != calls[1][1]
    rows = [
        json.loads(line) for line in (tmp_path / "catalog/studies.jsonl").open(encoding="utf-8")
    ]
    assert len(rows) == 1 and len(rows[0]["discovery_observations"]) == 2
    assert rows[0]["capabilities"]["species"]["status"] == "unknown"
    assert rows[0]["independent_units"] is None
    discover(plan(), tmp_path, resume=True, offline=True, _fetcher=fetch)
    assert len(calls) == 2


def test_failure_is_retained_and_only_explicit_retry_replaces_it(tmp_path):
    def mixed(job, cache, offline):
        return (
            {"status": "SOURCE_FAILURE", "records": [], "error": "offline miss"}
            if job["source"] == "geo"
            else result(job, cache, offline)
        )

    assert discover(plan(), tmp_path, _fetcher=mixed)["failed_source_jobs"] == 1
    discover(
        plan(), tmp_path, resume=True, _fetcher=lambda *args: pytest.fail("Must reuse failure")
    )
    final = discover(plan(), tmp_path, resume=True, retry_failed=True, _fetcher=result)
    assert final["failed_source_jobs"] == 0
    assert len(list((tmp_path / "attempt_history").glob("*.json"))) == 1


def test_changed_plan_and_corrupt_checkpoint_fail_closed(tmp_path):
    discover(plan(), tmp_path, _fetcher=result)
    changed = plan()
    changed["queries"][0]["query"] = "different"
    with pytest.raises(ValueError, match="same query plan"):
        discover(changed, tmp_path, resume=True)
    checkpoint = next((tmp_path / "checkpoints").glob("*.json"))
    checkpoint.write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        discover(plan(), tmp_path, resume=True)


def test_existing_unrelated_output_is_preserved(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("protected")
    with pytest.raises(ValueError, match="empty output"):
        discover(plan(), tmp_path, _fetcher=result)
    assert source.read_text() == "protected"


@pytest.mark.parametrize("change", ["unsafe", "duplicate", "unknown", "zero", "malformed"])
def test_invalid_plan_rejected(change):
    value = plan()
    row = value["queries"][0]
    if change == "unsafe":
        row["id"] = "../escape"
    elif change == "duplicate":
        value["queries"].append(dict(row))
    elif change == "unknown":
        row["dataset_sources"] = ["does-not-exist"]
    elif change == "zero":
        row["limit"] = 0
    else:
        value["queries"] = [None]
    with pytest.raises(ValueError):
        validate_plan(value)
