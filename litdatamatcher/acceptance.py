"""Strict machine acceptance validation for LitDataMatcher v2.

The acceptance report is deliberately derived from an evidence ledger rather
than accepted as an input.  A ledger check is usable only when it is tied to a
schema-valid, successful ``RUN_MANIFEST.json`` and to a successful command
whose retained log and declared output artifacts still match their SHA-256
digests.  This makes a README, an empty file, or a hand-written PASS report
insufficient evidence.

The ledger format is intentionally small and versioned.  It belongs beside
the immutable run evidence, outside source control when it references large
data.  See :func:`validate_acceptance` for the public entry point.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

JsonObject = dict[str, Any]

_TEMPLATE_DIR = Path(__file__).resolve().parent / "schemas_v2"
ACCEPTANCE_SCHEMA_PATH = _TEMPLATE_DIR / "ACCEPTANCE_REPORT.schema.json"
ACCEPTANCE_TEMPLATE_PATH = _TEMPLATE_DIR / "ACCEPTANCE_REPORT.template.json"
RUN_MANIFEST_SCHEMA_PATH = _TEMPLATE_DIR / "RUN_MANIFEST.schema.json"

IMPLEMENTATION_VERSION = "acceptance-validator-v2"
PRODUCT_GATE_IDS = tuple(f"G{number:02d}" for number in range(1, 17))
OPERATION_IDS = tuple(f"O{number:02d}" for number in range(1, 6))

# Every listed kind is a separately recorded executed observation.  The
# validator does not treat an arbitrary file, prose assertion, or coverage
# count as a substitute for any of these requirements.
GATE_REQUIREMENTS: dict[str, frozenset[str]] = {
    "G01": frozenset(
        {"clean_install", "new_document_input", "topic_input", "explicit_question_input"}
    ),
    "G02": frozenset(
        {
            "literature_coverage",
            "identifier_validation",
            "source_snapshot",
            "fulltext_parse",
            "duplicate_version_accounting",
        }
    ),
    "G03": frozenset(
        {
            "accession_study_coverage",
            "geo_path",
            "sequencing_repository_path",
            "mirror_deduplication",
        }
    ),
    "G04": frozenset(
        {
            "sample_profile",
            "processed_file_inspection",
            "feature_sample_alignment",
            "unit_counts",
            "usable_contrast",
        }
    ),
    "G05": frozenset(
        {
            "fresh_application_process",
            "previously_unprocessed_input",
            "runtime_model_revision",
            "no_prewritten_or_regex_substitution",
        }
    ),
    "G06": frozenset(
        {
            "claim_schema",
            "negation_direction_context",
            "quote_support",
            "entity_ambiguity",
            "locator_provenance_persistence",
        }
    ),
    "G07": frozenset(
        {
            "automatic_gap_generation",
            "user_question_mode",
            "answered_case",
            "partial_case",
            "contradictory_case",
            "insufficient_coverage_case",
        }
    ),
    "G08": frozenset(
        {
            "essential_requirements",
            "missing_vs_incompatible",
            "no_fit_case",
            "partial_fit_case",
            "joint_observation_constraint",
        }
    ),
    "G09": frozenset(
        {
            "external_resource_query",
            "dependence_contradiction_indirect_tests",
            "integration_mode_tests",
            "real_numeric_harmonization",
            "invalid_combination_abstention",
        }
    ),
    "G10": frozenset(
        {
            "baseline_hybrid_compatibility_comparison",
            "label_provenance",
            "study_grouped_holdout",
            "hard_negative",
            "source_disjoint_test",
            "opportunity_review",
            "score_explanation",
            "heuristic_labeling",
        }
    ),
    "G11": frozenset(
        {
            "source_update_invalidation",
            "idempotence",
            "offline_replay",
            "offline_fresh_inference",
            "no_hidden_download",
        }
    ),
    "G12": frozenset(
        {
            "two_stage_resume",
            "no_duplicate_or_lost_artifacts",
            "transient_error_handling",
            "schema_drift_handling",
            "corruption_handling",
            "inference_failure_handling",
        }
    ),
    "G13": frozenset(
        {
            "overlapping_jobs",
            "shared_writer_integrity",
            "resource_backoff",
            "task_owned_cleanup",
            "numeric_exits",
            "bounded_logs",
        }
    ),
    "G14": frozenset(
        {
            "independent_review",
            "reproduced_findings",
            "no_unresolved_high_functional_issue",
            "substantive_worker_pass",
            "integrated_refinement_round",
        }
    ),
    "G15": frozenset(
        {
            "stored_output_report",
            "traceable_claims",
            "escape_safe_text",
            "secret_license_injection_checks",
            "command_false_success_test",
        }
    ),
    "G16": frozenset(
        {
            "clean_delivery",
            "source_integrity",
            "version_lock_manifest_notice",
            "final_real_run",
            "machine_readiness_agreement",
            "status_stop_reason_agreement",
        }
    ),
}

OPERATION_REQUIREMENTS: dict[str, frozenset[str]] = {
    "O01": frozenset(
        {"source_preservation", "baseline_record", "worker_isolation", "lead_only_integration"}
    ),
    "O02": frozenset(
        {
            "safe_checkpoint",
            "remote_ref_match",
            "main_protection",
            "secret_bulk_data_check",
            "push_backlog_visibility",
        }
    ),
    "O03": frozenset(
        {"live_state", "continuation_command", "owner_lease", "pause_capacity_handling"}
    ),
    "O04": frozenset(
        {
            "supervisor_local_access",
            "supervisor_healthy_noop",
            "supervisor_deliberate_pause",
            "supervisor_abandoned_repair",
            "supervisor_stale_ownership",
            "supervisor_takeover_prevention",
            "supervisor_real_resume",
        }
    ),
    "O05": frozenset({"delivery_owner_stop_conditions", "completion_supervisor_disabled_or_idle"}),
}

REFINEMENT_KINDS = frozenset(
    {
        "substantive_worker_pass",
        "integrated_refinement_round",
        "no_material_gain_round",
        "untouched_holdout_pass",
    }
)

LEDGER_SCHEMA: JsonObject = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "build_id", "runs"],
    "properties": {
        "schema_version": {"const": "1.0"},
        "build_id": {"type": ["string", "null"]},
        "runs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["run_manifest", "checks"],
                "properties": {
                    "run_manifest": {"type": "string", "minLength": 1},
                    "checks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "id",
                                "target",
                                "kind",
                                "command_index",
                                "artifacts",
                                "observed_at",
                            ],
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                                "target": {"enum": list(PRODUCT_GATE_IDS + OPERATION_IDS)},
                                "kind": {"type": "string", "minLength": 1},
                                "command_index": {"type": "integer", "minimum": 0},
                                "artifacts": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "string", "minLength": 1},
                                },
                                "observed_at": {"type": "string", "format": "date-time"},
                                "subject": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
            },
        },
        "open_issues": {"type": "array"},
        "optional_backlog": {"type": "array", "items": {"type": "string"}},
        "stop_reason": {"type": ["string", "null"]},
    },
}


def validate_acceptance(
    evidence_path: str | Path | None,
    *,
    output_path: str | Path | None = None,
) -> JsonObject:
    """Validate a v1 evidence ledger and derive an acceptance report.

    ``evidence_path`` may be absent.  In that case the function emits the
    schema-valid all-``NOT_RUN`` report rather than pretending a run occurred.
    If ``output_path`` is supplied, the report is written atomically after it
    has passed the finalized acceptance-report schema.
    """

    report = _new_report()
    report["generated_at"] = _now()
    report["validator"] = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "command": "litdatamatcher-v2 acceptance",
        "executed": True,
    }
    if evidence_path is None:
        _set_all_not_run(report, "No acceptance evidence ledger was supplied.")
        return _finish(report, output_path)

    ledger_path = Path(evidence_path).resolve()
    if not ledger_path.is_file():
        _set_all_not_run(report, f"Acceptance evidence ledger is absent: {ledger_path}")
        return _finish(report, output_path)

    try:
        ledger = _read_json(ledger_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _set_all_fail(report, f"Acceptance evidence ledger is unreadable: {exc}")
        return _finish(report, output_path)

    ledger_errors = _schema_errors(LEDGER_SCHEMA, ledger)
    if ledger_errors:
        _set_all_fail(
            report, "Acceptance evidence ledger schema invalid: " + "; ".join(ledger_errors)
        )
        return _finish(report, output_path)

    report["build_id"] = ledger["build_id"]
    root = ledger_path.parent
    checks_by_target: dict[str, list[JsonObject]] = defaultdict(list)
    validated_runs: list[JsonObject] = []
    faults: list[str] = []
    for run_entry in ledger["runs"]:
        validated, run_faults = _validate_run_entry(root, run_entry)
        faults.extend(run_faults)
        if validated is None:
            continue
        validated_runs.append(validated)
        for check in validated["checks"]:
            checks_by_target[check["target"]].append(check)

    report["evidence_fingerprint"] = _fingerprint(validated_runs)
    report["coverage"] = _coverage(validated_runs)
    _apply_gate_results(report, checks_by_target, faults)
    _apply_operation_results(report, checks_by_target, faults)
    _apply_refinement(report, checks_by_target)
    report["open_issues"] = _valid_open_issues(ledger.get("open_issues", []), faults)
    report["optional_backlog"] = ledger.get("optional_backlog", [])
    report["stop_reason"] = ledger.get("stop_reason")
    report["calibration_status"], report["calibration_evidence"] = _calibration(validated_runs)
    report["automation_status"] = _automation_status(report)
    report["product_status"] = _product_status(report)
    return _finish(report, output_path)


def _validate_run_entry(root: Path, entry: JsonObject) -> tuple[JsonObject | None, list[str]]:
    faults: list[str] = []
    manifest_path = _safe_relative_path(root, entry["run_manifest"])
    if manifest_path is None or not manifest_path.is_file():
        return None, [f"run manifest is missing or escapes evidence root: {entry['run_manifest']}"]
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, [f"run manifest is unreadable ({entry['run_manifest']}): {exc}"]
    errors = _schema_errors(_read_json(RUN_MANIFEST_SCHEMA_PATH), manifest)
    if errors:
        return None, [
            f"run manifest schema invalid ({entry['run_manifest']}): " + "; ".join(errors)
        ]
    if manifest["execution_status"] != "PASS":
        return None, [
            f"run manifest is not PASS ({entry['run_manifest']}): {manifest['execution_status']}"
        ]
    if manifest["failures"]:
        return None, [f"run manifest records failures ({entry['run_manifest']})"]
    if not _valid_hex(manifest["source"]["working_tree_digest"], 64):
        return None, [f"run manifest has no valid source digest ({entry['run_manifest']})"]
    if not _valid_hex(manifest["source"]["config_digest"], 64):
        return None, [f"run manifest has no valid configuration digest ({entry['run_manifest']})"]

    run_root = manifest_path.parent
    artifacts: dict[str, JsonObject] = {}
    for artifact in manifest["artifacts"]:
        path = artifact["path"]
        safe = _safe_relative_path(run_root, path)
        if safe is None or not safe.is_file():
            faults.append(f"declared artifact missing or unsafe: {path}")
            continue
        actual_size = safe.stat().st_size
        actual_hash = _sha256(safe)
        if actual_size != artifact["size_bytes"] or actual_hash != artifact["sha256"]:
            faults.append(f"artifact digest mismatch: {path}")
            continue
        if artifact["validation"] != "PASS":
            faults.append(f"artifact is not validated PASS: {path}")
            continue
        artifacts[path] = artifact

    usable_checks: list[JsonObject] = []
    for check in entry["checks"]:
        check_faults = _validate_check(check, manifest, artifacts, run_root)
        if check_faults:
            faults.extend(f"{check['target']}/{check['id']}: {fault}" for fault in check_faults)
        else:
            usable_checks.append(check)
    return {"manifest": manifest, "manifest_path": manifest_path, "checks": usable_checks}, faults


def _validate_check(
    check: JsonObject, manifest: JsonObject, artifacts: dict[str, JsonObject], run_root: Path
) -> list[str]:
    faults: list[str] = []
    required = GATE_REQUIREMENTS.get(
        check["target"], OPERATION_REQUIREMENTS.get(check["target"], frozenset())
    )
    allowed = required | (REFINEMENT_KINDS if check["target"] == "G14" else frozenset())
    if check["kind"] not in allowed:
        return [f"unrecognized evidence kind {check['kind']!r}"]
    command_index = check["command_index"]
    if command_index >= len(manifest["commands"]):
        return ["command index is absent from run manifest"]
    command = manifest["commands"][command_index]
    if command["exit_code"] != 0:
        return [f"command exit code is not zero: {command['exit_code']}"]
    observed = _parse_time(check["observed_at"])
    started = _parse_time(manifest["started_at"])
    finished = _parse_time(manifest["finished_at"])
    if (
        observed is None
        or started is None
        or finished is None
        or not started <= observed <= finished
    ):
        return ["observed_at is outside the successful run interval; stale evidence is refused"]
    log_reference = command["log_reference"]
    if log_reference not in artifacts:
        return ["command log is not a hashed, validated manifest artifact"]
    log_path = _safe_relative_path(run_root, log_reference)
    if log_path is None or not log_path.is_file() or log_path.stat().st_size == 0:
        return ["command log is missing or empty"]
    for artifact_path in check["artifacts"]:
        if artifact_path not in artifacts:
            faults.append(
                f"check artifact is not a hashed, validated manifest artifact: {artifact_path}"
            )
    if not faults and not _has_semantic_attestation(check, run_root):
        faults.append(
            "no referenced structured audit attests this exact target/kind as PASS"
        )
    return faults


def _has_semantic_attestation(check: JsonObject, run_root: Path) -> bool:
    """Require machine-readable meaning in addition to a valid file digest.

    A check cannot be promoted merely by pointing at an arbitrary nonempty JSON
    file.  At least one referenced artifact must be a closeout audit containing
    the same target and evidence kind with a substantive PASS observation.
    """
    for artifact_path in check["artifacts"]:
        candidate = _safe_relative_path(run_root, artifact_path)
        if candidate is None or candidate.suffix.lower() != ".json":
            continue
        try:
            payload = _read_json(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        runner = payload.get("runner")
        evidence = payload.get("evidence")
        if not isinstance(runner, dict) or not str(
            runner.get("implementation_version", "")
        ).startswith("closeout-evidence-audit-"):
            continue
        if not isinstance(evidence, list):
            continue
        if any(
            isinstance(item, dict)
            and item.get("target") == check["target"]
            and item.get("kind") == check["kind"]
            and item.get("status") == "PASS"
            and isinstance(item.get("reason"), str)
            and item["reason"].strip()
            and isinstance(item.get("artifacts"), list)
            and item["artifacts"]
            for item in evidence
        ):
            return True
    return False


def _apply_gate_results(
    report: JsonObject, checks: dict[str, list[JsonObject]], faults: list[str]
) -> None:
    for gate, required in GATE_REQUIREMENTS.items():
        found = {check["kind"] for check in checks.get(gate, [])}
        gate_faults = [fault for fault in faults if fault.startswith(f"{gate}/")]
        if gate_faults:
            report["gates"][gate].update(status="FAIL", evidence=[], reason="; ".join(gate_faults))
        elif not found:
            report["gates"][gate].update(
                status="NOT_RUN", evidence=[], reason="No validated executed evidence."
            )
        else:
            missing = sorted(required - found)
            if missing:
                report["gates"][gate].update(
                    status="NOT_RUN",
                    evidence=_check_evidence(checks[gate]),
                    reason="Validated evidence is incomplete; missing: " + ", ".join(missing),
                )
            else:
                report["gates"][gate].update(
                    status="PASS",
                    evidence=_check_evidence(checks[gate]),
                    reason="All required executed evidence validated.",
                )


def _apply_operation_results(
    report: JsonObject, checks: dict[str, list[JsonObject]], faults: list[str]
) -> None:
    for operation, required in OPERATION_REQUIREMENTS.items():
        found = {check["kind"] for check in checks.get(operation, [])}
        op_faults = [fault for fault in faults if fault.startswith(f"{operation}/")]
        if op_faults:
            report["operations"][operation] = {"status": "FAIL", "evidence": []}
        elif not found:
            report["operations"][operation] = {"status": "NOT_RUN", "evidence": []}
        elif required - found:
            report["operations"][operation] = {
                "status": "NOT_RUN",
                "evidence": _check_evidence(checks[operation]),
            }
        else:
            report["operations"][operation] = {
                "status": "PASS",
                "evidence": _check_evidence(checks[operation]),
            }


def _apply_refinement(report: JsonObject, checks: dict[str, list[JsonObject]]) -> None:
    gate_checks = checks.get("G14", [])
    worker_counts = Counter(
        check.get("subject", "")
        for check in gate_checks
        if check["kind"] == "substantive_worker_pass"
    )
    worker_counts.pop("", None)
    round_ids = {
        check.get("subject", check["id"])
        for check in gate_checks
        if check["kind"] == "integrated_refinement_round"
    }
    no_gain_ids = {
        check.get("subject", check["id"])
        for check in gate_checks
        if check["kind"] == "no_material_gain_round"
    }
    report["refinement"] = {
        "minimum_worker_passes_observed": min(worker_counts.values()) if worker_counts else 0,
        "integrated_rounds": len(round_ids),
        "consecutive_no_material_gain_rounds": len(no_gain_ids),
        "untouched_holdout_pass": any(
            check["kind"] == "untouched_holdout_pass" for check in gate_checks
        ),
        "independent_reviews_pass": any(
            check["kind"] == "independent_review" for check in gate_checks
        ),
    }


def _fingerprint(runs: list[JsonObject]) -> JsonObject:
    if not runs:
        return {
            key: None
            for key in (
                "source_commit",
                "source_digest",
                "model_config_digest",
                "input_manifest_digest",
                "evaluation_protocol",
            )
        }
    manifests = [entry["manifest"] for entry in runs]
    source_commits = {item["source"]["commit"] for item in manifests if item["source"]["commit"]}
    source_digests = {
        item["source"]["working_tree_digest"]
        for item in manifests
        if item["source"]["working_tree_digest"]
    }
    protocols = {
        item["evaluation"]["protocol_version"]
        for item in manifests
        if item["evaluation"]["protocol_version"]
    }
    return {
        "source_commit": next(iter(source_commits)) if len(source_commits) == 1 else None,
        "source_digest": next(iter(source_digests)) if len(source_digests) == 1 else None,
        "model_config_digest": _digest_json([item["models"] for item in manifests]),
        "input_manifest_digest": _digest_json([_sha256(entry["manifest_path"]) for entry in runs]),
        "evaluation_protocol": next(iter(protocols)) if len(protocols) == 1 else None,
    }


def _coverage(runs: list[JsonObject]) -> JsonObject:
    keys = _new_report()["coverage"].keys()
    return {
        key: max((entry["manifest"]["coverage"][key] for entry in runs), default=0) for key in keys
    }


def _calibration(runs: list[JsonObject]) -> tuple[str, list[str]]:
    origins = {
        origin for entry in runs for origin in entry["manifest"]["evaluation"]["label_origins"]
    }
    evidence = [
        str(entry["manifest_path"])
        for entry in runs
        if entry["manifest"]["evaluation"]["label_origins"]
    ]
    if "expert" in origins:
        return "EXPERT_CALIBRATED", evidence
    if "source_determined" in origins or "model_assisted" in origins:
        return "SOURCE_ASSISTED_EVALUATION", evidence
    return "PENDING_EXPERT_LABELS", []


def _automation_status(report: JsonObject) -> str:
    if (
        report["operations"]["O05"]["status"] == "PASS"
        and report["gates"]["G16"]["status"] == "PASS"
        and report["stop_reason"]
    ):
        return "DISABLED_AT_COMPLETION"
    o04 = report["operations"]["O04"]["status"]
    if o04 == "PASS":
        return "VERIFIED_ENABLED"
    if o04 == "UNAVAILABLE":
        return "UNAVAILABLE"
    return "PREPARED_NOT_ENABLED"


def _product_status(report: JsonObject) -> str:
    gate_status = {gate: report["gates"][gate]["status"] for gate in PRODUCT_GATE_IDS}
    functional_gates = {"G01", "G05", "G06", "G07", "G08", "G15"}
    functional_floors = {
        "unique_literature_records": 50,
        "parsed_full_texts": 20,
        "unique_accession_studies": 50,
        "sample_profiled_studies": 20,
        "inspected_processed_studies": 1,
        "external_structured_resources": 1,
    }
    hardened_floors = {
        **functional_floors,
        "unique_literature_records": 200,
        "parsed_full_texts": 50,
        "unique_accession_studies": 100,
        "sample_profiled_studies": 30,
        "inspected_processed_studies": 2,
        "distinct_pilot_contexts": 2,
        "case_dossiers": 6,
    }
    if (
        all(value == "PASS" for value in gate_status.values())
        and _at_least(report["coverage"], hardened_floors)
        and report["refinement"]["minimum_worker_passes_observed"] >= 2
        and report["refinement"]["integrated_rounds"] >= 3
        and report["refinement"]["consecutive_no_material_gain_rounds"] >= 2
        and report["refinement"]["untouched_holdout_pass"]
        and report["refinement"]["independent_reviews_pass"]
        and report["stop_reason"]
        and report["evidence_fingerprint"]["source_commit"]
        and report["evidence_fingerprint"]["source_digest"]
        and not any(
            issue["severity"] in {"critical", "high"} and issue["affects_declared_functionality"]
            for issue in report["open_issues"]
        )
    ):
        return "HARDENED_ALPHA_READY"
    if all(gate_status[gate] == "PASS" for gate in functional_gates) and _at_least(
        report["coverage"], functional_floors
    ):
        return "FUNCTIONAL_ALPHA_AVAILABLE"
    return "NOT_READY"


def _valid_open_issues(value: Any, faults: list[str]) -> list[JsonObject]:
    # Reuse the finalized schema rather than accepting untyped commentary.
    errors = _schema_errors(
        {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "severity", "affects_declared_functionality", "description"],
                "properties": {
                    "id": {"type": "string"},
                    "severity": {"enum": ["critical", "high", "medium", "low"]},
                    "affects_declared_functionality": {"type": "boolean"},
                    "description": {"type": "string"},
                },
            },
        },
        value,
    )
    if errors:
        faults.append("open_issues schema invalid: " + "; ".join(errors))
        return [
            {
                "id": "acceptance-ledger-schema",
                "severity": "high",
                "affects_declared_functionality": True,
                "description": "Acceptance evidence ledger contains malformed open issues.",
            }
        ]
    return value


def _new_report() -> JsonObject:
    return copy.deepcopy(_read_json(ACCEPTANCE_TEMPLATE_PATH))


def _set_all_not_run(report: JsonObject, reason: str) -> None:
    for gate in PRODUCT_GATE_IDS:
        report["gates"][gate].update(status="NOT_RUN", evidence=[], reason=reason)
    for operation in OPERATION_IDS:
        report["operations"][operation] = {"status": "NOT_RUN", "evidence": []}


def _set_all_fail(report: JsonObject, reason: str) -> None:
    for gate in PRODUCT_GATE_IDS:
        report["gates"][gate].update(status="FAIL", evidence=[], reason=reason)
    for operation in OPERATION_IDS:
        report["operations"][operation] = {"status": "FAIL", "evidence": []}


def _finish(report: JsonObject, output_path: str | Path | None) -> JsonObject:
    errors = _schema_errors(_read_json(ACCEPTANCE_SCHEMA_PATH), report)
    if errors:
        raise RuntimeError("validator generated an invalid acceptance report: " + "; ".join(errors))
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)
    return report


def _schema_errors(schema: JsonObject, value: Any) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [error.message for error in sorted(validator.iter_errors(value), key=str)]


def _safe_relative_path(root: Path, value: str) -> Path | None:
    supplied = Path(value)
    if supplied.is_absolute() or ".." in supplied.parts:
        return None
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _read_json(path: Path) -> JsonObject:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON object expected")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _check_evidence(checks: list[JsonObject]) -> list[str]:
    return [f"{check['kind']}:{check['id']}" for check in checks]


def _at_least(values: JsonObject, floors: dict[str, int]) -> bool:
    return all(values[key] >= floor for key, floor in floors.items())


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _valid_hex(value: Any, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
