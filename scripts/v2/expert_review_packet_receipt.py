"""Write a deterministic receipt for the V2.4 blinded-review packet contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from litdatamatcher.data_plane import atomic_json
from litdatamatcher.expert_review import agreement_and_adjudication, build_blinded_review_packet, validate_review_labels


def build_receipt() -> dict:
    record = {
        "match_id": "source-determined-fixture-match",
        "label_origin": "source_determined",
        "question": {"question": "Does the source-described exposure affect the reported outcome?", "source_ids": ["PMID:fixture"], "evidence": [{"source_locator": "fixture:question:span", "text": "Source-determined fixture span."}]},
        "dataset": {"dataset_id": "GSE-fixture", "title": "Source-described fixture dataset", "source": "GEO", "organisms": ["Homo sapiens"], "assay_types": ["RNA-seq"]},
        "score": 1.0,
        "rank": 1,
    }
    built = build_blinded_review_packet([record], ["assigned-reviewer-a", "assigned-reviewer-b"])
    packet = built["packet"]
    validation = validate_review_labels(packet, [])
    agreement = agreement_and_adjudication(validation["valid_labels"])
    item = packet["items"][0]
    return {
        "schema_version": "v2_4_expert_review_receipt_v1",
        "fixture_label_origin": "source_determined",
        "review_status": packet["review_status"],
        "packet_id": packet["packet_id"],
        "item_count": len(packet["items"]),
        "source_span_count": len(item["question_source_spans"]),
        "assignment_count": packet["assignment_count"],
        "expert_label_count": len(validation["valid_labels"]),
        "adjudication_record_count": len(agreement["adjudication_records"]),
        "masking_validation": "PASS" if "score" not in item and "rank" not in item and item["question_source_spans"] else "FAIL",
        "validation_status": "PASS" if packet["review_status"] == "PENDING_EXPERT_REVIEW" and not validation["valid_labels"] else "FAIL",
        "limitation": "No expert labels, adjudication outcome, calibration, or gold-standard claim is present.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    atomic_json(args.out, build_receipt())


if __name__ == "__main__":
    main()
