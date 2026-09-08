"""Write a deterministic V2.3 evidence-compiler contract receipt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from litdatamatcher.data_plane import atomic_json, digest
from litdatamatcher.scientific_v2 import compile_evidence


def build_receipt() -> dict:
    question = {
        "question_id": "v2_3_receipt_question",
        "proposition_id": "v2_3_receipt_proposition",
        "conditions": {"organism": "human", "assay": "bulk_transcriptomics"},
        "gap_status": "unresolved-in-searched-coverage",
    }
    paper = {
        "evidence_id": "paper", "proposition_id": question["proposition_id"],
        "role": "perturbational_observation", "direction": "supports",
        "source_id": "PMID:receipt", "publication_id": "PMID:receipt",
        "study_id": "GSE:receipt", "conditions": question["conditions"],
        "measurement_type": "observation", "scope_match": "exact", "answers_question": True,
        "publication_date": "2026-09-01", "source_locator": "fixture:paper",
    }
    repository = {
        "evidence_id": "repository", "proposition_id": question["proposition_id"],
        "role": "metadata", "direction": "supports", "source_id": "GSE:receipt",
        "study_id": "GSE:receipt", "conditions": question["conditions"],
        "measurement_type": "metadata", "scope_match": "exact", "answers_question": False,
        "publication_date": "2026-09-01", "source_locator": "fixture:repository",
        "relation_assertions": [{
            "target_evidence_id": "paper", "relation_type": "derivative_evidence",
            "source_locator": "fixture:repository:linked-publication",
        }],
    }
    kg = {
        "evidence_id": "kg", "related_proposition_id": question["proposition_id"],
        "role": "curation", "direction": "supports", "source_id": "KG:receipt",
        "source_of_source": "GSE:receipt", "conditions": question["conditions"],
        "measurement_type": "curation", "scope_match": "related", "answers_question": False,
        "publication_date": "2026-09-01", "source_locator": "fixture:kg",
    }
    bundle = compile_evidence(question, [paper, repository, kg], "2026-09-08", [{"source": "fixture", "status": "success"}])
    relation_types = {edge["relation_type"] for edge in bundle["relation_graph"]["edges"]}
    return {
        "schema_version": "v2_3_evidence_compiler_receipt_v1",
        "fixture_scope": "synthetic contract fixture; no scientific finding or independent replication claim",
        "input_digest": digest([question, paper, repository, kg]),
        "bundle_id": bundle["bundle_id"],
        "gap_status": bundle["gap_status"],
        "evidence_item_count": len(bundle["evidence_items"]),
        "dependence_group_count": len(bundle["dependence_groups"]),
        "known_dependence_edge_count": bundle["known_dependence_edge_count"],
        "independent_support_count": bundle["independent_support_count"],
        "relation_types": sorted(relation_types),
        "validation_status": "PASS" if {"derivative_evidence", "same_underlying_evidence"} <= relation_types and len(bundle["dependence_groups"]) == 1 and bundle["independent_support_count"] is None else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    atomic_json(args.out, build_receipt())


if __name__ == "__main__":
    main()
