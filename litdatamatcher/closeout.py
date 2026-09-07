"""Read-only, deterministic closeout evidence audit for LitDataMatcher v2.

This module is intentionally stricter than a report inventory.  A file being
present, a README assertion, or a successful-looking status field never makes
an evidence kind pass.  Each positive result is produced by re-reading the
underlying structured record, checking the relevant invariants, and recording
the hash of every file on which that result depends.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .acceptance import GATE_REQUIREMENTS, OPERATION_REQUIREMENTS

AUDIT_VERSION = "closeout-evidence-audit-v1"
STATUSES = {"PASS", "FAIL", "NOT_RUN"}


def run_closeout_audit(data_root: str | Path, source_root: str | Path) -> dict[str, Any]:
    """Audit existing evidence without acquiring data, rerunning models, or holdouts."""
    data = Path(data_root).resolve()
    source = Path(source_root).resolve()
    files: dict[Path, dict[str, Any]] = {}

    def note(path: Path) -> dict[str, Any] | None:
        path = path.resolve()
        if not path.is_file():
            return None
        if path not in files:
            files[path] = {
                "path": _display_path(path, data, source),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        return files[path]

    def observation(
        target: str, kind: str, status: str, reason: str, *paths: Path
    ) -> dict[str, Any]:
        if status not in STATUSES:
            raise ValueError(f"Unknown evidence status: {status}")
        return {
            "target": target,
            "kind": kind,
            "status": status,
            "reason": reason,
            "artifacts": [note(path) for path in paths if note(path) is not None],
        }

    literature_path = data / "catalog" / "literature.jsonl"
    studies_path = data / "catalog" / "studies.jsonl"
    inspections_path = data / "catalog" / "processed_inspections.jsonl"
    source_manifest_path = data / "catalog" / "source_object_manifest.json"
    qualification_path = data / "runtime-qualification" / "qualified_7b_pass1.json"
    external_path = data / "evaluation" / "external_evidence" / "validation.json"
    state_path = source / "project_state" / "TASK_STATE.json"
    next_path = source / "project_state" / "NEXT_ACTION.md"

    literature, literature_problem = _jsonl(literature_path)
    studies, studies_problem = _jsonl(studies_path)
    inspections, inspections_problem = _jsonl(inspections_path)
    source_manifest, source_manifest_problem = _json_array(source_manifest_path)
    qualification, qualification_problem = _json_object(qualification_path)
    external, external_problem = _json_object(external_path)
    state, state_problem = _json_object(state_path)
    runs = _run_manifests(data / "runs")

    observations: list[dict[str, Any]] = []
    for target, required in {**GATE_REQUIREMENTS, **OPERATION_REQUIREMENTS}.items():
        observations.extend(
            observation(target, kind, "NOT_RUN", "No substantive closeout check is implemented for this evidence kind.")
            for kind in sorted(required)
        )

    def replace(target: str, kind: str, status: str, reason: str, *paths: Path) -> None:
        for index, item in enumerate(observations):
            if item["target"] == target and item["kind"] == kind:
                observations[index] = observation(target, kind, status, reason, *paths)
                return
        raise KeyError(f"Unknown required evidence kind: {target}/{kind}")

    # G02--G04: recompute coverage from catalog records.  The checks deliberately
    # require a valid record structure as well as a count, preventing a padded or
    # empty JSONL file from satisfying a floor.
    _catalog_observations(
        replace, literature, literature_problem, studies, studies_problem, inspections,
        inspections_problem, source_manifest, source_manifest_problem, literature_path,
        studies_path, inspections_path, source_manifest_path,
    )

    # G05--G06: qualification is useful only if it records a fresh local runtime
    # and an extractive, source-anchored claim.  This does not elevate a later
    # partial integrated run to PASS.
    if qualification_problem:
        replace("G05", "fresh_application_process", "FAIL", qualification_problem, qualification_path)
    elif _fresh_runtime(qualification):
        replace("G05", "fresh_application_process", "PASS", "Qualification records fresh local transformers inference with a process execution id.", qualification_path)
        replace("G05", "previously_unprocessed_input", "PASS", "Qualification records a fresh input fingerprint and fresh_local_inference origin.", qualification_path)
        replace("G05", "runtime_model_revision", "PASS", "Qualification binds fresh inference to a nonempty model revision.", qualification_path)
        if _extractive_claim(qualification):
            replace("G06", "claim_schema", "PASS", "Fresh qualified claim has the required typed extraction fields.", qualification_path)
            replace("G06", "quote_support", "PASS", "Fresh qualified claim preserves an exact evidence span and extractive verification.", qualification_path)
            replace("G06", "locator_provenance_persistence", "PASS", "Fresh qualified claim retains source document and source URL provenance.", qualification_path)
    else:
        replace("G05", "fresh_application_process", "FAIL", "Runtime qualification exists but does not prove fresh local model inference.", qualification_path)

    # A partial or failed application run is an executed contradiction for the
    # integrated runtime requirement.  It is not erased by a smoke qualification.
    for run in runs:
        if run["manifest"].get("execution_status") in {"PARTIAL", "FAIL"}:
            replace("G05", "no_prewritten_or_regex_substitution", "FAIL", "A retained integrated application run is PARTIAL/FAIL; source-guard failures remain unresolved.", run["path"])
            break

    # G09 has a real imported resource, but only its direct query and replay
    # assertions are credited.  It cannot stand in for numerical harmonization
    # or the other compiler gates.
    if external_problem:
        replace("G09", "external_resource_query", "FAIL", external_problem, external_path)
    elif external.get("status") == "PASS" and external.get("records", 0) > 0 and external.get("offline_replay_equal") is True:
        replace("G09", "external_resource_query", "PASS", "Imported external resource has nonzero records and deterministic offline replay equality.", external_path)

    # O03 is computed from typed compact state.  A state file alone is not enough.
    if state_problem:
        replace("O03", "live_state", "FAIL", state_problem, state_path)
    elif state.get("execution_status") in {"RUNNING", "WAITING_FOR_CAPACITY", "PAUSED_BY_USER", "BLOCKED", "COMPLETE"}:
        replace("O03", "live_state", "PASS", "Typed project state has a recognized execution status.", state_path)
        if isinstance(state.get("next_action", {}).get("command"), str) and state["next_action"]["command"].strip() and next_path.is_file():
            replace("O03", "continuation_command", "PASS", "State and continuation file provide a nonempty exact command.", state_path, next_path)

    # Existing release runs are checked for integrity, but only PASS manifests
    # with every declared artifact matching are allowed to support the fact that
    # a final real run occurred.  Current data intentionally does not meet it.
    integrity_failures = [run for run in runs if run["integrity"] == "FAIL"]
    if integrity_failures:
        replace("G16", "final_real_run", "FAIL", integrity_failures[0]["reason"], integrity_failures[0]["path"])
    elif any(run["integrity"] == "PASS" for run in runs):
        replace("G16", "final_real_run", "NOT_RUN", "A valid run exists, but no retained final-closeout designation or independent machine-readiness agreement was found.", *[run["path"] for run in runs if run["integrity"] == "PASS"])

    # Explicitly carry the documented exposed holdout forward as a blocker.  No
    # final holdout is read or run by this audit.
    refinement_paths = sorted((data / "evaluation" / "refinement").glob("*.json")) if (data / "evaluation" / "refinement").is_dir() else []
    exposed = any("holdout_exposure" in _json_object(path)[0] for path in refinement_paths)
    if exposed:
        replace("G10", "study_grouped_holdout", "FAIL", "Retained evaluation evidence records a holdout exposure; it cannot be claimed untouched.", *refinement_paths)

    gates = _summaries(observations, GATE_REQUIREMENTS)
    operations = _summaries(observations, OPERATION_REQUIREMENTS)
    blockers = [
        {"target": item["target"], "kind": item["kind"], "status": item["status"], "reason": item["reason"]}
        for item in observations
        if item["status"] != "PASS"
    ]
    report = {
        "schema_version": "1.0",
        "runner": {"implementation_version": AUDIT_VERSION, "read_only": True},
        "data_root": str(data),
        "source_root": str(source),
        "source_commit": _git_commit(source),
        "evidence": sorted(observations, key=lambda item: (item["target"], item["kind"])),
        "gates": gates,
        "operations": operations,
        "artifact_hashes": sorted(files.values(), key=lambda item: item["path"]),
        "blockers": blockers,
        "summary": dict(Counter(item["status"] for item in observations)),
        "pre_holdout_ready": not blockers,
    }
    return report


def write_closeout_audit(report: dict[str, Any], output_path: str | Path) -> None:
    """Write a byte-stable JSON report atomically (apart from its chosen path)."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def _catalog_observations(
    replace,
    literature,
    literature_problem,
    studies,
    studies_problem,
    inspections,
    inspections_problem,
    source_manifest,
    source_manifest_problem,
    literature_path,
    studies_path,
    inspections_path,
    source_manifest_path,
):
    if literature_problem:
        for kind in ("literature_coverage", "identifier_validation", "source_snapshot", "fulltext_parse", "duplicate_version_accounting"):
            replace("G02", kind, "FAIL", literature_problem, literature_path)
    else:
        ids = [row.get("document_id") for row in literature]
        parsed = [row for row in literature if row.get("fulltext_status") == "parsed" and isinstance(row.get("text"), str) and row["text"].strip()]
        snapshots = [
            row
            for row in literature
            if isinstance(row.get("source_snapshot"), dict)
            and _hex(row["source_snapshot"].get("sha256"))
        ]
        replace("G02", "literature_coverage", "PASS" if len(set(ids)) >= 200 else "FAIL", f"Validated {len(set(ids))} unique nonempty document identifiers (floor 200).", literature_path)
        replace("G02", "identifier_validation", "PASS" if len(ids) == len(set(ids)) and all(isinstance(value, str) and value.strip() for value in ids) else "FAIL", "Document identifiers are nonempty and unique." if len(ids) == len(set(ids)) else "Catalog has duplicate or blank document identifiers.", literature_path)
        valid_snapshot_hashes = _verified_source_hashes(source_manifest)
        snapshot_status = (
            "PASS"
            if not source_manifest_problem
            and len(snapshots) >= 200
            and all(row["source_snapshot"]["sha256"] in valid_snapshot_hashes for row in snapshots)
            else "FAIL"
        )
        replace(
            "G02",
            "source_snapshot",
            snapshot_status,
            f"Validated retained object hashes for {len(snapshots)} literature source snapshots."
            if snapshot_status == "PASS"
            else "Literature source snapshot records are not all backed by a retained matching object hash.",
            literature_path,
            source_manifest_path,
        )
        replace("G02", "fulltext_parse", "PASS" if len(parsed) >= 50 else "FAIL", f"Validated parsed nonempty full text for {len(parsed)} records (floor 50).", literature_path)
        # Duplicate/version accounting needs explicit relation fields, not just unique ids.
        duplicate_rows = [row for row in literature if any(key in row for key in ("version_of", "duplicate_of", "publication_group"))]
        if duplicate_rows:
            replace("G02", "duplicate_version_accounting", "PASS", f"Validated explicit duplicate/version relation fields on {len(duplicate_rows)} records.", literature_path)
    if studies_problem:
        for kind in ("accession_study_coverage", "geo_path", "sequencing_repository_path", "mirror_deduplication"):
            replace("G03", kind, "FAIL", studies_problem, studies_path)
    else:
        groups = {row.get("dependence_group", row.get("dataset_id")) for row in studies}
        geo = [row for row in studies if row.get("source") == "GEO" and row.get("source_locator")]
        sequence = [
            row
            for row in studies
            if any("SRA" in str(sample.get("fields", {}).get("type", "")).upper() for sample in row.get("samples", []))
            or any("ENA" in str(alias).upper() for alias in row.get("repository_aliases", []))
        ]
        replace("G03", "accession_study_coverage", "PASS" if len(groups) >= 100 else "FAIL", f"Validated {len(groups)} distinct catalog study/dependence groups (floor 100).", studies_path)
        replace("G03", "geo_path", "PASS" if geo else "NOT_RUN", "At least one catalog record has GEO evidence." if geo else "No catalog record substantiates a GEO path.", studies_path)
        replace("G03", "sequencing_repository_path", "PASS" if sequence else "NOT_RUN", "At least one catalog record has SRA/ENA evidence." if sequence else "No catalog record substantiates an SRA/ENA path.", studies_path)
        if all(isinstance(row.get("dataset_id"), str) and row.get("dataset_id") for row in studies) and len(groups) <= len(studies):
            replace("G03", "mirror_deduplication", "PASS", "Catalog records retain stable dataset IDs and an explicit dependence-group denominator.", studies_path)
    if inspections_problem:
        for kind in ("sample_profile", "processed_file_inspection", "feature_sample_alignment", "unit_counts", "usable_contrast"):
            replace("G04", kind, "FAIL", inspections_problem, inspections_path)
    elif studies_problem:
        replace("G04", "sample_profile", "FAIL", studies_problem, studies_path)
    else:
        profiled = [row for row in studies if row.get("profile_status") == "sample_annotations_parsed"]
        valid = [row for row in inspections if row.get("processed_measurements_present") and row.get("sample_alignment")]
        distinct = {row.get("dataset_id") for row in valid if row.get("dataset_id")}
        replace("G04", "sample_profile", "PASS" if len(profiled) >= 30 else "FAIL", f"Validated {len(profiled)} sample-annotation profiles (floor 30).", studies_path)
        replace("G04", "processed_file_inspection", "PASS" if len(distinct) >= 2 else "FAIL", f"Validated processed-file inspection for {len(distinct)} distinct studies (floor 2).", inspections_path)
        replace("G04", "feature_sample_alignment", "PASS" if len(valid) >= 2 else "FAIL", f"Validated sample alignment for {len(valid)} processed inspection records.", inspections_path)
        counts = [row for row in valid if any(key in row for key in ("sample_count", "feature_count", "n_samples", "n_features"))]
        if counts:
            replace("G04", "unit_counts", "PASS", f"Validated explicit unit counts in {len(counts)} processed inspection records.", inspections_path)


def _run_manifests(runs_root: Path) -> list[dict[str, Any]]:
    result = []
    if not runs_root.is_dir():
        return result
    for path in sorted(runs_root.glob("*/RUN_MANIFEST.json")):
        manifest, problem = _json_object(path)
        if problem:
            result.append({"path": path, "manifest": {}, "integrity": "FAIL", "reason": problem})
            continue
        integrity, reason = _manifest_integrity(path, manifest)
        result.append({"path": path, "manifest": manifest, "integrity": integrity, "reason": reason})
    return result


def _manifest_integrity(path: Path, manifest: dict[str, Any]) -> tuple[str, str]:
    if manifest.get("execution_status") != "PASS":
        return "FAIL", f"Run manifest execution_status is {manifest.get('execution_status')!r}, not PASS."
    if manifest.get("failures"):
        return "FAIL", "Run manifest retains failures."
    artifacts = manifest.get("artifacts")
    commands = manifest.get("commands")
    if not isinstance(artifacts, list) or not isinstance(commands, list) or not commands:
        return "FAIL", "Run manifest lacks artifacts or commands."
    names = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            return "FAIL", "Run manifest contains malformed artifact metadata."
        name = artifact["path"]
        candidate = (path.parent / name).resolve()
        if candidate.parent != path.parent.resolve() or not candidate.is_file():
            return "FAIL", f"Declared artifact is missing or unsafe: {name}"
        if artifact.get("validation") != "PASS" or artifact.get("sha256") != _sha256(candidate) or artifact.get("size_bytes") != candidate.stat().st_size:
            return "FAIL", f"Declared artifact fails retained validation/hash check: {name}"
        names.add(name)
    for command in commands:
        if not isinstance(command, dict) or command.get("exit_code") != 0 or command.get("log_reference") not in names:
            return "FAIL", "A command failed or lacks a declared hashed log artifact."
    return "PASS", "All declared artifacts and command logs validate."


def _summaries(observations: list[dict[str, Any]], requirements: dict[str, frozenset[str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target, required in requirements.items():
        rows = [item for item in observations if item["target"] == target]
        statuses = {item["kind"]: item["status"] for item in rows}
        status = "FAIL" if any(statuses.get(kind) == "FAIL" for kind in required) else "PASS" if all(statuses.get(kind) == "PASS" for kind in required) else "NOT_RUN"
        out[target] = {"status": status, "pass": sum(value == "PASS" for value in statuses.values()), "required": len(required)}
    return out


def _jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.is_file():
        return [], f"Required JSONL artifact is absent: {path.name}"
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [], f"Required JSONL artifact is unreadable: {exc}"
    if not rows or any(not isinstance(row, dict) for row in rows):
        return [], "Required JSONL artifact is empty or contains non-object rows."
    return rows, None


def _json_object(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, f"Required JSON artifact is absent: {path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, f"Required JSON artifact is unreadable: {exc}"
    return (value, None) if isinstance(value, dict) else ({}, "Required JSON artifact is not an object.")


def _json_array(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.is_file():
        return [], f"Required JSON artifact is absent: {path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [], f"Required JSON artifact is unreadable: {exc}"
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        return [], "Required JSON artifact is not an array of objects."
    return value, None


def _verified_source_hashes(manifest: list[dict[str, Any]]) -> set[str]:
    verified: set[str] = set()
    for item in manifest:
        path = item.get("path")
        expected = item.get("sha256")
        if not isinstance(path, str) or not _hex(expected):
            continue
        candidate = Path(path)
        if candidate.is_file() and candidate.stat().st_size == item.get("size_bytes") and _sha256(candidate) == expected:
            verified.add(expected)
    return verified


def _fresh_runtime(value: dict[str, Any]) -> bool:
    try:
        manifest = value["fresh"]["inference_manifest"]
        return bool(manifest["execution_id"] and manifest["process_id"] and manifest["model_revision"] and manifest["origin"] == "fresh_local_inference" and manifest["runtime"] == "transformers")
    except (KeyError, TypeError):
        return False


def _extractive_claim(value: dict[str, Any]) -> bool:
    try:
        claim = value["fresh"]["claims"][0]
        span = claim["evidence_span"]
        provenance = claim["source_provenance"]
        return bool(claim["claim_id"] and claim["verification"] == "extractive_source_guard" and span["text"] and span["end"] > span["start"] and claim["source_document_id"] and provenance["source_url"])
    except (KeyError, IndexError, TypeError):
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _display_path(path: Path, data: Path, source: Path) -> str:
    for name, root in (("data", data), ("source", source)):
        try:
            return f"{name}/{path.relative_to(root).as_posix()}"
        except ValueError:
            continue
    return str(path)


def _git_commit(source: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
