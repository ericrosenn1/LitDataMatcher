"""Write a deterministic receipt for bounded adapter pagination semantics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from litdatamatcher.adapters import _bounded_cursor_pages
from litdatamatcher.data_plane import atomic_json, digest


class FixtureClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.last_response_metadata = {"cache_status": "synthetic_fixture"}

    def get_json(self, url, params=None):
        return self.payloads.pop(0)


def pages(payloads, max_pages=4):
    return _bounded_cursor_pages(
        FixtureClient(payloads),
        "https://fixture.invalid/paged",
        {"query": "fixture"},
        extract_items=lambda payload: payload.get("items"),
        cursor_field="next",
        cursor_param="cursor",
        max_pages=max_pages,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    complete = pages([{"items": [{"id": "one"}], "next": "two"}, {"items": [{"id": "two"}]}])
    repeated = pages([{"items": [], "next": "again"}, {"items": [], "next": "again"}])
    truncated = pages([{"items": [], "next": "two"}, {"items": [], "next": "three"}], max_pages=2)
    drift = pages([{"items": "schema-drift"}])
    class ErrorClient:
        def get_json(self, url, params=None):
            raise RuntimeError("fixture failure")
    error = _bounded_cursor_pages(ErrorClient(), "https://fixture.invalid/paged", {"query": "fixture"}, extract_items=lambda payload: payload.get("items"), cursor_field="next", cursor_param="cursor")
    statuses = {"complete": complete["pagination"]["status"], "repeated": repeated["pagination"]["status"], "truncated": truncated["pagination"]["status"], "schema_drift": drift["pagination"]["status"], "error": error["pagination"]["status"]}
    partial = [repeated, truncated, drift, error]
    passed = statuses == {"complete": "COMPLETE", "repeated": "REPEATED_CURSOR", "truncated": "TRUNCATED_PAGE_LIMIT", "schema_drift": "SCHEMA_DRIFT", "error": "ERROR"} and complete["pagination"]["candidate_universe_status"] == "COMPLETE_CANDIDATE_UNIVERSE" and all(item["pagination"]["candidate_universe_status"] == "PARTIAL_CANDIDATE_UNIVERSE_NOT_EVIDENCE_COMPLETE" for item in partial)
    atomic_json(args.out, {"schema_version": "v2_adapter_pagination_receipt_v1", "fixture_scope": "synthetic/local cursor pages only", "statuses": statuses, "complete_request_scopes": complete["pagination"]["pages"], "input_digest": digest({"complete": complete, "repeated": repeated, "truncated": truncated, "drift": drift}), "validation_status": "PASS" if passed else "FAIL"})


if __name__ == "__main__":
    main()
