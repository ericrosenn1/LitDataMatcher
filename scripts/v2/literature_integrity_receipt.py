"""Write a deterministic synthetic V2 literature-integrity receipt."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from litdatamatcher.data_plane import atomic_json
from litdatamatcher.literature_integrity import consolidate_literature_rows
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True); args = parser.parse_args()
    row = {"source_id": "pubmed:fixture", "source": "pubmed", "source_provenance": {"source_type": "pubmed", "retrieval_time_utc": "2026-09-08T00:00:00Z"}, "version_relationships": {"is-correction-of": [{"id": "old"}]}, "metadata": {"alternate_source_ids": ["crossref:fixture"]}}
    result = consolidate_literature_rows([row], [{"source": "pubmed", "status": "OBSERVED"}, {"source": "crossref", "status": "UNKNOWN_RETRIEVAL_OR_SCHEMA_FAILURE"}])[0]["metadata"]["literature_integrity"]
    atomic_json(args.out, {"schema_version": "v2_literature_integrity_receipt_v1", "fixture_scope": "synthetic metadata only", "lifecycle_status": result["lifecycle_status"], "fulltext_status": result["fulltext_status"], "source_statuses": result["source_statuses"], "evidence_eligibility": result["evidence_eligibility"], "validation_status": "PASS" if result["lifecycle_status"] == "CORRECTED_REQUIRES_VERSION_REVIEW" and result["fulltext_status"] == "UNKNOWN" else "FAIL"})
if __name__ == "__main__": main()
