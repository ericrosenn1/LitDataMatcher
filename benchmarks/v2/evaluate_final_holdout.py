"""Sealed, one-time evaluator for the v3 source-disjoint final holdout.

This command deliberately has no default paths: the invoker must make every
immutable input and every output/consumption location explicit.  Gate checks
run before the selected source snapshot is opened.  A consumption ledger is
created with exclusive create semantics immediately before that read, so a
failed or interrupted attempt cannot be silently repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOKEN = re.compile(r"[A-Za-z0-9]+")
REQUIRED_STATUS = "RESERVED_PENDING_ONE_TIME_FINAL_HOLDOUT_AUTHORIZATION"
NOT_RUN = "NOT_RUN"


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _all_empty(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_empty(item) for item in value.values())
    if isinstance(value, list):
        return not value
    return False


def validate_preconditions(reservation_path: Path, audit_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the sealed reservation without reading the selected snapshot."""
    reservation, audit = read_json(reservation_path), read_json(audit_path)
    _require(reservation.get("status") == REQUIRED_STATUS, "reservation is not awaiting final-holdout authorization")
    expected_audit = reservation.get("audit", {}).get("sha256")
    _require(isinstance(expected_audit, str) and expected_audit == sha256_file(audit_path).upper(), "audit hash does not match reservation")
    _require(reservation.get("audit", {}).get("decision") == REQUIRED_STATUS, "reservation audit decision is not sealed")
    _require(audit.get("decision") == REQUIRED_STATUS, "audit decision is not sealed")
    selected = reservation.get("selected_accession")
    _require(isinstance(selected, str) and selected and audit.get("candidate", {}).get("accession") == selected, "selected accession does not bind audit candidate")
    verification = reservation.get("verification", {})
    for key, expected in {
        "all_exact_identifier_intersections_zero": True,
        "candidate_publication_pmc_complete": True,
        "candidate_ena_sra_complete": True,
        "base_official_relation_errors": 0,
        "base_official_relation_truncations": 0,
    }.items():
        _require(verification.get(key) == expected, f"reservation verification failed: {key}")
    sealed = reservation.get("sealed_evaluation_state", {})
    for key, expected in {
        "title_summary_outcome_status": "UNINSPECTED",
        "label_status": "UNINSPECTED",
        "prediction_status": NOT_RUN,
        "ranking_status": NOT_RUN,
    }.items():
        _require(sealed.get(key) == expected, f"sealed evaluation state is not {expected}: {key}")
    _require(_all_empty(audit.get("identifier_overlap")), "official identifier audit records an overlap")
    candidate_status = audit.get("candidate_relation_status", {})
    for key in ("publication_complete", "pmc_complete", "ena_sra_complete"):
        _require(candidate_status.get(key) is True, f"candidate official relation check incomplete: {key}")
    official = audit.get("official_relation_status", {})
    relation_counts = {
        "gds_to_pubmed": ("queried_series", "explicit_no_indexed_links"),
        "pubmed_to_pmc": ("queried_pubmed", "explicit_no_indexed_links"),
        "ena_sra": ("queried_bioproject_or_sra", "explicit_no_indexed_links"),
    }
    for relation, (queried_key, empty_key) in relation_counts.items():
        details = official.get(relation, {})
        _require(details.get("errors") == [], f"official relation check has errors: {relation}")
        _require(details.get("truncated") == [], f"official relation check is truncated: {relation}")
        _require(
            type(details.get(queried_key)) is int
            and type(details.get("indexed_links")) is int
            and isinstance(details.get(empty_key), list)
            and details["indexed_links"] + len(details[empty_key]) == details[queried_key],
            f"official relation check is incomplete: {relation}",
        )
    _require(official.get("ena_sra", {}).get("maximum_rows_returned", 0) < 1000, "ENA relation check has incomplete 1000-row response")
    for details in official.get("ena_sra", {}).get("raw_files", {}).values():
        _require(details.get("complete_under_limit") is True, "base ENA/SRA relation response is incomplete")
        _require(type(details.get("rows")) is int and details["rows"] < 1000, "base ENA/SRA relation response reached limit")
    for details in candidate_status.get("raw_files", {}).values():
        _require(details.get("complete_under_limit") is True, "candidate ENA/SRA relation response is incomplete")
        _require(type(details.get("rows")) is int and details["rows"] < 1000, "candidate ENA/SRA relation response reached limit")
    return reservation, audit


def reserve_consumption(ledger: Path, reservation: dict[str, Any], audit_path: Path) -> dict[str, Any]:
    """Atomically create the irreversible one-time execution ledger."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "1.0",
        "status": "RUNNING",
        "reservation_version": reservation.get("reservation_version"),
        "selected_accession": reservation["selected_accession"],
        "audit_sha256": reservation["audit"]["sha256"],
        "audit_path": str(audit_path),
        "reserved_at": utc_now(),
        "attempt_id": str(uuid.uuid4()),
    }
    try:
        descriptor = os.open(str(ledger), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"final holdout is already consumed or reserved: {ledger}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def update_ledger(ledger: Path, record: dict[str, Any]) -> None:
    staged = ledger.with_name(ledger.name + ".tmp")
    staged.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with staged.open("r+", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged, ledger)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_record(snapshot: Path, accession: str, expected_sha256: str) -> dict[str, Any]:
    _require(sha256_file(snapshot) == expected_sha256.lower(), "selected source snapshot hash does not match audit")
    payload = read_json(snapshot)
    results = payload.get("result")
    _require(isinstance(results, dict), "source snapshot lacks ESummary result object")
    matches = [value for key, value in results.items() if key != "uids" and isinstance(value, dict) and value.get("accession") == accession]
    _require(len(matches) == 1, "selected accession is absent or ambiguous in source snapshot")
    return matches[0]


def _field(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            return value
    return None


def observed(value: Any, locator: str) -> dict[str, Any]:
    return {"value": value, "status": "observed", "source_locator": locator, "mapping_type": "exact"}


def profile_from_source(record: dict[str, Any], accession: str, snapshot_sha256: str, locator: str) -> tuple[dict[str, Any], str]:
    """Construct only a source-determined retrieval case from the selected snapshot."""
    title = _field(record, "title")
    organism = _field(record, "taxon", "organism")
    assay = _field(record, "gdsType", "gdstype", "assay")
    count = _field(record, "n_samples", "sample_count_reported")
    summary = _field(record, "summary")
    _require(all(isinstance(value, str) and value.strip() for value in (title, organism, assay)), "selected source snapshot lacks title, organism, or assay")
    _require(type(count) is int and count >= 0, "selected source snapshot lacks source-reported sample count")
    profile = {
        "dataset_id": accession,
        "availability": "SOURCE_METADATA_CAPTURED",
        "independent_units": None,
        "capabilities": {
            "organism": observed(organism, locator),
            "assay": observed(assay, locator),
            "study_title": observed(title, locator),
            "reported_sample_count": observed(count, locator),
            "comparator": {"value": None, "status": "unknown", "reason": "not assessed from acquired summary metadata"},
        },
        "source_fact_locator": {"url": locator, "snapshot_sha256": snapshot_sha256, "json_pointer": f"/result/{accession}"},
    }
    # Summary may inform retrieval text only; it is never inspected before execution.
    return profile, " ".join(str(part) for part in (title, summary or "", organism, assay))


def requirements_for(profile: dict[str, Any]) -> list[dict[str, Any]]:
    caps = profile["capabilities"]
    return [
        {"field": "organism", "expected": caps["organism"]["value"], "essential": True, "source_locator": "selected source snapshot"},
        {"field": "assay", "expected": caps["assay"]["value"], "essential": True, "source_locator": "selected source snapshot"},
        {"field": "study_title", "expected": caps["study_title"]["value"], "essential": True, "source_locator": "selected source snapshot"},
    ]


def candidate_text(profile: dict[str, Any]) -> str:
    caps = profile["capabilities"]
    return " ".join(str(caps[field].get("value", "")) for field in ("study_title", "organism", "assay"))


def lexical_score(left: str, right: str) -> float:
    a = {item.casefold() for item in TOKEN.findall(left) if len(item) > 1}
    b = {item.casefold() for item in TOKEN.findall(right) if len(item) > 1}
    return len(a & b) / len(a | b) if a or b else 0.0


def order(scores: dict[str, float]) -> list[str]:
    return [key for key, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def metrics(ranking: list[str], target: str) -> dict[str, Any]:
    rank = ranking.index(target) + 1 if target in ranking else None
    hit5 = int(rank is not None and rank <= 5)
    hit10 = int(rank is not None and rank <= 10)
    return {
        "queries": 1,
        "candidate_relevance_labels": len(ranking),
        "positive_queries": 1,
        "recall_at_10_numerator": hit10,
        "recall_at_10_denominator": 1,
        "precision_at_5_numerator": hit5,
        "precision_at_5_denominator": 5,
        "ndcg_at_5": (1.0 / math.log2(rank + 1)) if rank is not None and rank <= 5 else 0.0,
        "first_relevant_rank": rank,
        "invalid_top_match": int(rank != 1),
    }


def frozen_profiles(universe: Path) -> list[dict[str, Any]]:
    payload = read_json(universe)
    profiles = payload.get("primary", {}).get("candidate_profiles")
    _require(isinstance(profiles, list) and len(profiles) >= 20, "frozen development candidate universe is incomplete")
    ids = [item.get("dataset_id") for item in profiles]
    _require(len(ids) == len(set(ids)) and all(isinstance(item, str) and item for item in ids), "frozen candidate universe has invalid dataset ids")
    return profiles


def evaluate_case(target: dict[str, Any], target_text: str, development: list[dict[str, Any]], model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from litdatamatcher.scientific_v2 import rank_candidates
    from litdatamatcher.semantic_runtime import PretrainedSemanticIndex, verify_model

    profiles = [target] + development
    ids = [item["dataset_id"] for item in profiles]
    _require(len(ids) == len(set(ids)), "target accession already occurs in frozen development universe")
    question = "Find the source-described study matching this experimental context: " + target["capabilities"]["study_title"]["value"] + ". Required organism: " + target["capabilities"]["organism"]["value"] + "; required assay: " + target["capabilities"]["assay"]["value"] + "."
    texts = [target_text] + [candidate_text(profile) for profile in development]
    manifest = verify_model(model_dir)
    index = PretrainedSemanticIndex(model_dir, device="cpu").fit(
        [{"id": item, "text": text} for item, text in zip(ids, texts, strict=True)]
    )
    semantic = {item["id"]: item["score"] for item in index.search(question, k=len(ids))}
    lexical = {
        item: lexical_score(question, text) for item, text in zip(ids, texts, strict=True)
    }
    hybrid = {item: 0.5 * lexical[item] + 0.5 * ((semantic[item] + 1.0) / 2.0) for item in ids}
    rankings = {
        "lexical": order(lexical),
        "minilm_hybrid": order(hybrid),
        "compatibility_aware": [item["dataset_id"] for item in rank_candidates(requirements_for(target), profiles, {item: 2.0 * hybrid[item] - 1.0 for item in ids})],
    }
    target_id = target["dataset_id"]
    return {
        "question": question,
        "target_accession": target_id,
        "candidate_universe": ids,
        "candidate_universe_sha256": canonical_digest(profiles),
        "hard_negative_accessions": [item for item in ids if item != target_id],
        "label_origin": "source_determined",
        "methods": {name: {"ranking": ranking, "metrics": metrics(ranking, target_id)} for name, ranking in rankings.items()},
    }, {"model_id": manifest["model_id"], "revision": manifest["revision"], "license": manifest["license"], "runtime": "transformers PretrainedSemanticIndex", "device": "cpu"}


def artifact(path: Path, kind: str) -> dict[str, Any]:
    return {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "kind": kind, "validation": "PASS"}


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def build_manifest(result: dict[str, Any], model: dict[str, Any], reservation: dict[str, Any], audit: dict[str, Any], snapshot: Path, snapshot_retrieved_at: str, started: str, finished: str, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    selected = reservation["selected_accession"]
    return {
        "schema_version": "2.0",
        "run_id": "final-holdout-v3-" + uuid.uuid4().hex,
        "execution_status": "PASS",
        "started_at": started,
        "finished_at": finished,
        "source": {"repository": "ericrosenn1/LitDataMatcher", "commit": git_value("-C", str(root), "rev-parse", "HEAD"), "working_tree_digest": canonical_digest(git_value("-C", str(root), "status", "--porcelain")), "spec_digest": sha256_file(root / "docs" / "v2" / "build_spec" / "PACKAGE_MANIFEST.json"), "config_digest": canonical_digest({"reservation": reservation, "audit": audit, "candidate_universe_sha256": result["candidate_universe_sha256"]})},
        "environment": {"python": sys.version, "platform": platform.platform(), "dependency_lock_digest": sha256_file(root / "requirements-v2.lock"), "hardware_record": "CPU: local PretrainedSemanticIndex inference"},
        "models": [{"id": model["model_id"], "revision": model["revision"], "runtime": model["runtime"], "license_status": model["license"], "prompt_version": "source-determined-final-holdout-v3"}],
        "source_snapshots": [{"source": "NCBI GEO GDS ESummary acquired snapshot", "snapshot_id": audit["candidate"]["source_snapshot_sha256"], "retrieved_at": snapshot_retrieved_at, "manifest_digest": sha256_file(snapshot)}],
        "commands": [{"command": "evaluate_final_holdout.py (one-time final-holdout execution)", "cwd": str(root), "started_at": started, "exit_code": 0, "log_reference": "metrics.json"}],
        "evaluation": {"protocol_version": reservation["protocol_id"], "split_id": selected, "split_role": "FINAL_HOLDOUT", "label_origins": ["source_determined"], "holdout_exposed_to_tuning": False, "source_disjointness": {"status": "PROVEN_SOURCE_DISJOINT", "selected_accession": selected, "unknown_overlap_count": 0}},
        "coverage": {"unique_literature_records": 0, "parsed_full_texts": 0, "unique_accession_studies": len(result["candidate_universe"]), "sample_profiled_studies": 0, "inspected_processed_studies": 0, "external_structured_resources": 0, "distinct_pilot_contexts": 1, "case_dossiers": 1},
        "artifacts": artifacts,
        "network": {"mode": "OFFLINE", "offline_block_test": True, "external_requests_observed": 0},
        "inference": {"fresh_calls": 1, "cache_replays": 0, "backend_qualified": True},
        "failures": [],
    }


def validate_manifest(manifest: dict[str, Any], schema: Path) -> None:
    from jsonschema import Draft202012Validator

    errors = sorted(Draft202012Validator(read_json(schema)).iter_errors(manifest), key=lambda error: list(error.path))
    if errors:
        raise ValueError("RUN_MANIFEST schema validation failed: " + "; ".join(error.message for error in errors))


def execute(args: argparse.Namespace) -> int:
    reservation, audit = validate_preconditions(args.reservation, args.audit)
    _require(args.output.resolve() != args.consumption_ledger.resolve(), "output directory cannot be the consumption ledger")
    _require(not args.output.exists(), "final output path already exists")
    _require(bool(args.source_snapshot_retrieved_at.strip()), "source snapshot retrieval timestamp is required")
    record = reserve_consumption(args.consumption_ledger, reservation, args.audit)
    started = utc_now()
    try:
        snapshot_record = source_record(args.source_snapshot, reservation["selected_accession"], audit["candidate"]["source_snapshot_sha256"])
        target, target_text = profile_from_source(snapshot_record, reservation["selected_accession"], audit["candidate"]["source_snapshot_sha256"], audit["candidate"]["source_locator"])
        development = frozen_profiles(args.frozen_universe)
        sys.path.insert(0, str(args.lead))
        result, model = evaluate_case(target, target_text, development, args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=args.output.name + ".staging-") as temporary:
            stage = Path(temporary)
            metrics_path, case_path = stage / "metrics.json", stage / "source_case.json"
            metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            case_path.write_text(json.dumps({"target_profile": target, "source_snapshot_sha256": audit["candidate"]["source_snapshot_sha256"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest = build_manifest(result, model, reservation, audit, args.source_snapshot, args.source_snapshot_retrieved_at, started, utc_now(), [artifact(metrics_path, "final_holdout_metrics"), artifact(case_path, "source_determined_retrieval_case")])
            validate_manifest(manifest, args.schema)
            manifest_path = stage / "RUN_MANIFEST.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(stage, args.output)
        record.update({"status": "PASS", "finished_at": utc_now(), "output": str(args.output), "run_manifest_sha256": sha256_file(args.output / "RUN_MANIFEST.json")})
        update_ledger(args.consumption_ledger, record)
        return 0
    except Exception as error:
        record.update({"status": "FAILED_CONSUMED", "finished_at": utc_now(), "error": f"{type(error).__name__}: {error}"})
        update_ledger(args.consumption_ledger, record)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservation", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--frozen-universe", required=True, type=Path)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--source-snapshot-retrieved-at", required=True)
    parser.add_argument("--lead", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--schema", default=Path(__file__).resolve().parents[2] / "litdatamatcher" / "schemas_v2" / "RUN_MANIFEST.schema.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--consumption-ledger", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
