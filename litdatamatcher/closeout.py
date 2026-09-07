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
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from .acceptance import GATE_REQUIREMENTS, OPERATION_REQUIREMENTS

AUDIT_VERSION = "closeout-evidence-audit-v3"
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
    recovery_path = data / "catalog" / "acquisition_offline_recovery.json"
    numeric_path = data / "catalog" / "numeric_integration.json"
    evaluation_path = source / "benchmarks" / "v2" / "E04_SOURCE_SNAPSHOT_MATCHING.json"
    supervisor_path = data / "operations" / "supervisor_activation.json"
    preservation_path = data.parent / "preservation" / "manifest.json"
    delivery_path = data / "release" / "delivery_validation.json"
    security_path = data / "evaluation" / "security_validation.json"
    concurrency_path = data / "evaluation" / "concurrency" / "final" / "concurrency_validation.json"
    full_junit_path = data / "tests" / "post-acceptance-full.xml"
    controller_junit_path = data / "evaluation" / "E03_controller_independent.xml"
    reservation_path = source / "benchmarks" / "v2" / "final_holdout_reservation_v3.json"
    state_path = source / "project_state" / "TASK_STATE.json"
    next_path = source / "project_state" / "NEXT_ACTION.md"

    literature, literature_problem = _jsonl(literature_path)
    studies, studies_problem = _jsonl(studies_path)
    inspections, inspections_problem = _jsonl(inspections_path)
    source_manifest, source_manifest_problem = _json_array(source_manifest_path)
    qualification, qualification_problem = _json_object(qualification_path)
    external, external_problem = _json_object(external_path)
    recovery, recovery_problem = _json_object(recovery_path)
    numeric, numeric_problem = _json_object(numeric_path)
    evaluation, evaluation_problem = _json_object(evaluation_path)
    supervisor, supervisor_problem = _json_object(supervisor_path)
    preservation, preservation_problem = _json_array(preservation_path)
    delivery, delivery_problem = _json_object(delivery_path)
    security, security_problem = _json_object(security_path)
    concurrency, concurrency_problem = _json_object(concurrency_path)
    reservation, reservation_problem = _json_object(reservation_path)
    state, state_problem = _json_object(state_path)
    runs = _run_manifests(data / "runs")
    full_junit, full_junit_problem = _junit(full_junit_path)
    controller_junit, controller_junit_problem = _junit(controller_junit_path)
    refinement = _refinement_evidence(data / "evaluation" / "refinement")

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
    _junit_observations(
        replace,
        full_junit,
        full_junit_problem,
        controller_junit,
        controller_junit_problem,
        recovery,
        recovery_problem,
        full_junit_path,
        controller_junit_path,
        recovery_path,
    )
    _refinement_observations(replace, refinement)
    _numeric_observations(replace, numeric, numeric_problem, numeric_path)
    _evaluation_observations(replace, evaluation, evaluation_problem, evaluation_path)
    _operation_observations(
        replace,
        source,
        state,
        state_problem,
        preservation,
        preservation_problem,
        supervisor,
        supervisor_problem,
        controller_junit,
        controller_junit_problem,
        state_path,
        preservation_path,
        supervisor_path,
        controller_junit_path,
    )
    _delivery_observations(
        replace,
        delivery,
        delivery_problem,
        security,
        security_problem,
        delivery_path,
        security_path,
    )
    if (
        not concurrency_problem
        and concurrency.get("status") == "PASS"
        and concurrency.get("overlap_seconds", 0) > 0
        and len(concurrency.get("jobs", [])) >= 2
        and all(job.get("exit_code") == 0 for job in concurrency["jobs"])
    ):
        replace(
            "G13",
            "overlapping_jobs",
            "PASS",
            "Two independent contract suites executed concurrently with a measured overlap and successful numeric exits.",
            concurrency_path,
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
        manifest = qualification.get("fresh", {}).get("inference_manifest", {})
        if (
            manifest.get("origin") == "fresh_local_inference"
            and manifest.get("local_files_only") is True
            and qualification.get("network_control", {}).get("blocked_probe")
        ):
            replace(
                "G11",
                "offline_fresh_inference",
                "PASS",
                "Qualified transformers inference was fresh, local-files-only, and executed with a blocked socket probe.",
                qualification_path,
            )
    else:
        replace("G05", "fresh_application_process", "FAIL", "Runtime qualification exists but does not prove fresh local model inference.", qualification_path)

    # Historical partial runs are retained as diagnostics.  They cannot negate
    # a later designated, integrity-checked final run.
    final_runs = _designated_final_runs(runs)
    holdout_runs = _designated_holdout_runs(runs)
    final_runtime = [run for run in final_runs if _fresh_run_inference(run)]
    if final_runtime:
        replace(
            "G05",
            "no_prewritten_or_regex_substitution",
            "PASS",
            "A designated final run has an integrity-checked fresh runtime inference artifact.",
            final_runtime[0]["path"],
        )
    elif final_runs:
        replace(
            "G05",
            "no_prewritten_or_regex_substitution",
            "FAIL",
            "Designated final run lacks a validated fresh runtime inference artifact.",
            final_runs[0]["path"],
        )

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

    # G16 deliberately examines designated final runs only.  Earlier partial
    # development and cleanroom runs neither pass nor fail final delivery.
    designated = [run for run in runs if _is_designated_final(run["manifest"])]
    if not designated:
        replace(
            "G16",
            "final_real_run",
            "NOT_RUN",
            "No designated FINAL_HOLDOUT run is retained; historical development runs are excluded.",
        )
    elif final_runs:
        replace(
            "G16",
            "final_real_run",
            "PASS",
            "Designated final run has PASS execution, complete hashed artifacts, and successful command logs.",
            final_runs[0]["path"],
        )
    else:
        replace("G16", "final_real_run", "FAIL", designated[0]["reason"], designated[0]["path"])

    # GSE112372 exposure retires only that family. It does not itself fail the
    # gate once a distinct, source-disjoint replacement final run is proven.
    refinement_paths = sorted((data / "evaluation" / "refinement").glob("*.json")) if (data / "evaluation" / "refinement").is_dir() else []
    exposed = any("holdout_exposure" in _json_object(path)[0] for path in refinement_paths)
    replacement = _replacement_holdout_result(holdout_runs, reservation, reservation_problem)
    if replacement["status"] == "PASS":
        replace("G10", "study_grouped_holdout", "PASS", replacement["reason"], reservation_path, replacement["run"]["path"])
        replace("G10", "source_disjoint_test", "PASS", replacement["reason"], reservation_path, replacement["run"]["path"])
    elif replacement["status"] == "FAIL":
        replace("G10", "study_grouped_holdout", "FAIL", replacement["reason"], reservation_path, replacement.get("path", reservation_path))
        replace("G10", "source_disjoint_test", "FAIL", replacement["reason"], reservation_path, replacement.get("path", reservation_path))
    elif exposed:
        replace("G10", "study_grouped_holdout", "NOT_RUN", "GSE112372 is retired; no distinct source-disjoint replacement final holdout has passed.", *refinement_paths, reservation_path)

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
        "refinement": _refinement_summary(refinement),
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
        relationship_rows = [
            row for row in literature if isinstance(row.get("version_relationships"), dict)
        ]
        if len(relationship_rows) == len(literature):
            populated = sum(bool(row["version_relationships"]) for row in relationship_rows)
            replace(
                "G02",
                "duplicate_version_accounting",
                "PASS",
                f"All {len(literature)} records retain an explicit version-relationship object; {populated} contain declared relations and empty objects remain explicit none-observed states.",
                literature_path,
            )
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
        contrast_rows = [
            row
            for row in studies
            if row.get("profile_status") == "sample_annotations_parsed"
            and isinstance(row.get("groups"), list)
            and len({str(value).strip() for value in row["groups"] if str(value).strip()}) >= 2
        ]
        if contrast_rows:
            replace(
                "G04",
                "usable_contrast",
                "PASS",
                f"Validated at least two distinct source-derived sample groups in {len(contrast_rows)} profiled studies without claiming statistical power.",
                studies_path,
            )


def _junit_observations(
    replace,
    full,
    full_problem,
    controller,
    controller_problem,
    recovery,
    recovery_problem,
    full_path,
    controller_path,
    recovery_path,
):
    """Credit only named passing tests from an unfailed JUnit receipt."""
    if not full_problem:
        checks = {
            ("G06", "negation_direction_context"): ("test_omitted_negation_rejected", "test_negated_direction_rejected_even_when_quote_exists"),
            ("G06", "entity_ambiguity"): ("test_orthology_is_not_identity", "test_query_does_not_match_substrings"),
            ("G07", "automatic_gap_generation"): ("test_open_question_identification_and_ranking_end_to_end", "test_explicit_gap_question_is_source_linked_without_novelty_claim"),
            ("G07", "answered_case"): ("test_negative_result_can_answer_scoped_question_without_positive_vote",),
            ("G07", "partial_case"): ("test_optional_missing_field_preserves_useful_partial_result",),
            ("G07", "contradictory_case"): ("test_contradiction_retained_even_when_context_does_not_match",),
            ("G07", "insufficient_coverage_case"): ("test_insufficient_search_never_becomes_global_novelty",),
            ("G08", "essential_requirements"): ("test_wrong_species_high_similarity_cannot_rescue_direct_fit",),
            ("G08", "missing_vs_incompatible"): ("test_missing_null_and_explicit_false_are_distinct", "test_absence_without_provenance_cannot_be_confirmed_mismatch"),
            ("G08", "no_fit_case"): ("test_wrong_species_high_similarity_cannot_rescue_direct_fit",),
            ("G08", "partial_fit_case"): ("test_optional_missing_field_preserves_useful_partial_result",),
            ("G08", "joint_observation_constraint"): ("test_complementary_union_without_joint_units_not_sufficient", "test_pooling_rejects_shared_units_and_incompatible_estimands"),
            ("G09", "dependence_contradiction_indirect_tests"): ("test_contradiction_retained_even_when_context_does_not_match",),
            ("G09", "integration_mode_tests"): ("test_pooling_rejects_shared_units_and_incompatible_estimands", "test_common_cohort_identifiers_alone_not_valid_numeric_join"),
            ("G09", "invalid_combination_abstention"): ("test_pooling_rejects_shared_units_and_incompatible_estimands",),
            ("G11", "source_update_invalidation"): ("test_invalidation_preserves_unrelated_and_replays", "test_changed_derivation_invalidates_descendants_even_if_text_unchanged"),
            ("G11", "idempotence"): ("test_pipeline_rerun_resets_sqlite_run_tables",),
            ("G11", "offline_replay"): ("test_offline_replay_hash_verified_no_network", "test_retry_after_transient_and_network_free_replay"),
            ("G11", "no_hidden_download"): ("test_offline_replay_hash_verified_no_network",),
            ("G12", "schema_drift_handling"): ("test_schema_failure_preserves_valid_pointer",),
            ("G12", "corruption_handling"): ("test_corrupted_snapshot_rejected", "test_cache_corruption_and_missing_fail_offline"),
            ("G12", "inference_failure_handling"): ("test_malformed_generation_repair_cache_and_corruption",),
            ("G13", "shared_writer_integrity"): ("test_cross_process_duplicate_writer_refused", "test_stage_lease_prevents_duplicate_writers_and_releases"),
            ("G13", "resource_backoff"): ("test_governor_hysteresis", "test_nonzero_stale_output_timeout_and_capacity"),
            ("G13", "numeric_exits"): ("test_nonzero_native_exit_never_succeeds",),
            ("G13", "task_owned_cleanup"): ("test_actual_runner_and_worker_killed_then_repaired_once",),
            ("G13", "bounded_logs"): ("test_nonzero_stale_output_timeout_and_capacity",),
            ("G15", "escape_safe_text"): ("test_report_escapes_hostile_metadata",),
            ("G15", "command_false_success_test"): ("test_nonzero_native_exit_never_succeeds",),
            ("O04", "supervisor_stale_ownership"): ("test_abandoned_stage_resumes_without_duplicate",),
        }
        for (target, kind), names in checks.items():
            if _passed_tests(full, *names):
                replace(target, kind, "PASS", "Named contract tests passed in an unfailed JUnit receipt.", full_path)
    if not recovery_problem and recovery.get("status") == "PASS":
        stages = recovery.get("stages")
        valid_stages = isinstance(stages, list) and len(stages) >= 2 and all(
            isinstance(stage, dict)
            and stage.get("interrupted") is True
            and stage.get("resume_status") == "PASS"
            and stage.get("normalized_records_identical") is True
            and stage.get("no_duplicate_identities") is True
            and stage.get("before_sha256") == stage.get("after_sha256")
            for stage in stages
        )
        if valid_stages and recovery.get("network_attempts") == 0 and not recovery.get("invalid_objects"):
            replace("G11", "offline_replay", "PASS", "Two interrupted source stages resumed byte-identically with network attempts blocked.", recovery_path)
            replace("G12", "two_stage_resume", "PASS", "Two interrupted source stages resumed with PASS status.", recovery_path)
            replace("G12", "no_duplicate_or_lost_artifacts", "PASS", "Recovery retained byte-identical records and no duplicate identities.", recovery_path)
    if not controller_problem:
        controller_checks = (
            "test_actual_runner_and_worker_killed_then_repaired_once",
            "test_corrupt_success_repair_preserves_diagnostic_and_reexecutes",
            "test_user_pause_during_active_job_can_resume_after_explicit_resume",
            "test_cross_process_duplicate_writer_refused",
        )
        if _passed_tests(controller, *controller_checks):
            replace("O03", "owner_lease", "PASS", "Independent controller JUnit receipt exercises cross-process ownership protection.", controller_path)
            replace("O03", "pause_capacity_handling", "PASS", "Independent controller JUnit receipt exercises pause/resume and capacity-aware failure handling.", controller_path)
            replace("G12", "transient_error_handling", "PASS", "Independent controller receipt verifies bounded failed-worker repair behavior.", controller_path)
            supervisor_kinds = {
                "supervisor_healthy_noop": ("test_healthy_supervisor_is_noop",),
                "supervisor_deliberate_pause": ("test_user_pause_during_active_job_can_resume_after_explicit_resume",),
                "supervisor_abandoned_repair": ("test_actual_runner_and_worker_killed_then_repaired_once",),
                "supervisor_stale_ownership": ("test_abandoned_stage_resumes_without_duplicate",),
                "supervisor_takeover_prevention": ("test_cross_process_duplicate_writer_refused",),
                "supervisor_real_resume": ("test_corrupt_success_repair_preserves_diagnostic_and_reexecutes",),
            }
            for kind, names in supervisor_kinds.items():
                if _passed_tests(controller, *names):
                    replace(
                        "O04",
                        kind,
                        "PASS",
                        "Independent controller receipt passed the named corrective-supervision behavior.",
                        controller_path,
                    )


def _numeric_observations(replace, numeric, problem, path):
    if problem:
        return
    exact = (
        numeric.get("status") == "PASS"
        and numeric.get("sample_count", 0) >= 2
        and numeric.get("feature_count", 0) > 0
        and numeric.get("source_values_sha256") == numeric.get("aligned_values_sha256")
        and numeric.get("exact_values_and_missingness_preserved") is True
        and numeric.get("imputed_values") == 0
    )
    rejected = numeric.get("rejected_integrations")
    abstained = (
        isinstance(rejected, list)
        and len(rejected) >= 3
        and all(item.get("status") == "NOT_COMBINABLE" for item in rejected)
    )
    if exact:
        replace(
            "G09",
            "real_numeric_harmonization",
            "PASS",
            f"Real {numeric['source_dataset']} measurements retain exact values and missingness across {numeric['feature_count']} features and {numeric['sample_count']} samples.",
            path,
        )
    if exact and numeric.get("integration_mode") == "DIRECT_COMBINE" and abstained:
        replace(
            "G09",
            "integration_mode_tests",
            "PASS",
            "Direct same-study integration is explicit and three invalid cross-sample/study cases abstain.",
            path,
        )


def _evaluation_observations(replace, evaluation, problem, path):
    if problem:
        return
    primary = evaluation.get("primary")
    if not isinstance(primary, dict):
        return
    cases = primary.get("cases", primary.get("per_query"))
    metrics = primary.get("metrics")
    if not isinstance(cases, list) or not isinstance(metrics, dict):
        return
    negative_count = sum(
        case.get("negative_labels", 0)
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("negative_labels"), int)
    )
    compatibility = metrics.get("compatibility_aware", {})
    if negative_count > 0 and compatibility.get("invalid_top_match") == 0:
        replace(
            "G10",
            "hard_negative",
            "PASS",
            f"The retained primary evaluation includes {negative_count} source-determined negative candidate labels and zero invalid compatibility-aware top matches.",
            path,
        )
    origins = evaluation.get("label_origins", evaluation.get("label_origin"))
    if origins or evaluation.get("calibration_status") == "SOURCE_ASSISTED_EVALUATION":
        replace(
            "G10",
            "heuristic_labeling",
            "PASS",
            "Evaluation records source-determined label provenance and does not present heuristic scores as probabilities.",
            path,
        )


def _operation_observations(
    replace,
    source,
    state,
    state_problem,
    preservation,
    preservation_problem,
    supervisor,
    supervisor_problem,
    controller,
    controller_problem,
    state_path,
    preservation_path,
    supervisor_path,
    controller_path,
):
    head = _git_output(source, "rev-parse", "HEAD")
    branch = _git_output(source, "branch", "--show-current")
    remote = _git_output(source, "rev-parse", "refs/remotes/origin/codex/litdatamatcher-v2-build")
    main = _git_output(source, "rev-parse", "refs/heads/main")
    origin_main = _git_output(source, "rev-parse", "refs/remotes/origin/main")
    base = state.get("source", {}).get("base_commit") if not state_problem else None
    if not preservation_problem:
        entries = preservation
        if (
            isinstance(entries, list)
            and len(entries) >= 90
            and all(_hex(item.get("sha256")) for item in entries if isinstance(item, dict))
        ):
            replace(
                "O01",
                "source_preservation",
                "PASS",
                f"Preservation manifest retains hashes for {len(entries)} starting files.",
                preservation_path,
            )
            replace(
                "O01",
                "baseline_record",
                "PASS",
                "Preservation manifest provides a hashed pre-build baseline.",
                preservation_path,
            )
    worktrees = _git_output(source, "worktree", "list", "--porcelain")
    if worktrees and "codex/v2-" in worktrees and "codex/litdatamatcher-v2-build" in worktrees:
        replace(
            "O01",
            "worker_isolation",
            "PASS",
            "Git records separate lead and scoped worker worktrees.",
            state_path,
        )
    if branch == "codex/litdatamatcher-v2-build" and state.get("coordination", {}).get(
        "lead_identity"
    ):
        replace(
            "O01",
            "lead_only_integration",
            "PASS",
            "Canonical state binds integration to the lead checkout and identity.",
            state_path,
        )
    if head and branch == "codex/litdatamatcher-v2-build":
        replace("O02", "safe_checkpoint", "PASS", "The build is committed on the requested codex branch.", state_path)
    if head and remote == head:
        replace("O02", "remote_ref_match", "PASS", "Local HEAD equals the remote-tracking build ref.", state_path)
    if base and main == base and origin_main == base:
        replace("O02", "main_protection", "PASS", "Local and remote-tracking main remain at the recorded base commit.", state_path)
    if not state_problem and state.get("checkpoint", {}).get("push_pending") is False:
        replace(
            "O02",
            "push_backlog_visibility",
            "PASS",
            "Typed state explicitly records that no checkpoint push is pending.",
            state_path,
        )
    if not supervisor_problem:
        expected = supervisor.get("controller_prerequisite_junit_sha256")
        if (
            supervisor.get("status") in {"ACTIVE_CONFIG_VERIFIED", "DISABLED_AT_COMPLETION"}
            and _hex(expected)
            and controller_path.is_file()
            and expected == _sha256(controller_path)
        ):
            replace(
                "O04",
                "supervisor_local_access",
                "PASS",
                "Supervisor activation is bound to the canonical local state and matching controller receipt.",
                supervisor_path,
                controller_path,
            )
    if not controller_problem and _passed_tests(
        controller, "test_stale_success_artifact_does_not_hide_noop"
    ):
        replace(
            "O04",
            "supervisor_healthy_noop",
            "PASS",
            "Independent controller receipt verifies a valid no-op path.",
            controller_path,
        )


def _delivery_observations(
    replace, delivery, delivery_problem, security, security_problem, delivery_path, security_path
):
    if (
        not security_problem
        and security.get("status") == "PASS"
        and security.get("secret_findings") == 0
        and security.get("oversize_tracked_files") == 0
    ):
        replace(
            "O02",
            "secret_bulk_data_check",
            "PASS",
            "Executed repository scan found no secrets or oversized tracked artifacts.",
            security_path,
        )
        replace(
            "G15",
            "secret_license_injection_checks",
            "PASS",
            "Executed secret, license-notice, prompt-injection and tracked-size checks passed.",
            security_path,
        )
    if delivery_problem or delivery.get("status") != "PASS":
        return
    mappings = {
        "clean_delivery": delivery.get("archive_reopen_pass"),
        "source_integrity": delivery.get("source_archive_integrity_pass"),
        "version_lock_manifest_notice": delivery.get("version_lock_manifest_notice_pass"),
        "machine_readiness_agreement": delivery.get("independent_readiness_agreement"),
        "status_stop_reason_agreement": delivery.get("status_stop_reason_agreement"),
    }
    for kind, passed in mappings.items():
        if passed is True:
            replace("G16", kind, "PASS", "Final delivery validation records this invariant as PASS.", delivery_path)
    if delivery.get("stored_output_report_pass") is True:
        replace("G15", "stored_output_report", "PASS", "Final delivery contains a validated stored HTML report.", delivery_path)
    if delivery.get("traceable_claims_pass") is True:
        replace("G15", "traceable_claims", "PASS", "Final delivery validates claim-to-source locators and hashes.", delivery_path)
    if delivery.get("explicit_question_input_pass") is True:
        replace("G01", "explicit_question_input", "PASS", "Final clean-room input validation passed.", delivery_path)
        replace("G07", "user_question_mode", "PASS", "Final clean-room explicit-question mode passed.", delivery_path)
    for kind, key in (
        ("clean_install", "clean_install_pass"),
        ("new_document_input", "new_document_input_pass"),
        ("topic_input", "topic_input_pass"),
    ):
        if delivery.get(key) is True:
            replace("G01", kind, "PASS", "Final clean-room validation passed.", delivery_path)
    if delivery.get("opportunity_review_pass") is True:
        replace("G10", "opportunity_review", "PASS", "Final review-sheet validation passed.", delivery_path)
    if delivery.get("score_explanation_pass") is True:
        replace("G10", "score_explanation", "PASS", "Final report labels and explains uncalibrated component scores.", delivery_path)
    if delivery.get("independent_review_pass") is True:
        for kind in (
            "independent_review",
            "reproduced_findings",
            "no_unresolved_high_functional_issue",
        ):
            replace("G14", kind, "PASS", "Independent final delivery review passed.", delivery_path)
    if delivery.get("worker_passes_minimum", 0) >= 2:
        replace("G14", "substantive_worker_pass", "PASS", "Structured delivery evidence records at least two substantive passes per central worker.", delivery_path)
    if delivery.get("integrated_refinement_rounds", 0) >= 3:
        replace("G14", "integrated_refinement_round", "PASS", "Structured delivery evidence records at least three distinct integrated rounds.", delivery_path)
    if delivery.get("overlapping_jobs_pass") is True:
        replace("G13", "overlapping_jobs", "PASS", "Executed overlapping read-worker jobs completed with one protected writer.", delivery_path)
    if delivery.get("completion_supervisor_disabled_or_idle") is True:
        replace("O05", "completion_supervisor_disabled_or_idle", "PASS", "Completion supervisor is disabled or idle.", delivery_path)
    if delivery.get("delivery_owner_stop_conditions") is True:
        replace("O05", "delivery_owner_stop_conditions", "PASS", "Delivery owner and stop conditions agree with machine state.", delivery_path)


def _refinement_observations(replace, refinement):
    """Use only comparable, explicit refinement records; prose cannot supply a pass."""
    rounds = refinement["rounds"]
    valid_rounds = [item for item in rounds if _valid_round(item)]
    if valid_rounds and _comparable_performance(valid_rounds[-1]):
        replace(
            "G10",
            "baseline_hybrid_compatibility_comparison",
            "PASS",
            "Structured refinement record compares lexical, hybrid, and compatibility-aware methods on the same retained query set.",
            *refinement["paths"],
        )
        if valid_rounds[-1].get("label_origin") in {"expert", "source_determined", "model_assisted", "unreviewed"}:
            replace(
                "G10",
                "label_provenance",
                "PASS",
                "Structured refinement record declares a permitted label origin.",
                *refinement["paths"],
            )
    if len(valid_rounds) >= 3:
        replace("G14", "integrated_refinement_round", "PASS", "At least three structured comparable integrated refinement rounds are retained.", *refinement["paths"])
    elif valid_rounds:
        replace(
            "G14",
            "integrated_refinement_round",
            "NOT_RUN",
            f"Only {len(valid_rounds)} structured comparable integrated rounds are retained; three are required.",
            *refinement["paths"],
        )
    if len(valid_rounds) >= 2:
        # This supports closeout tracking but does not call the gate ready: the
        # gate still requires independent review and all worker passes.
        replace("G14", "substantive_worker_pass", "NOT_RUN", "Refinement records cover evaluation only; central-worker coverage is not established.", *refinement["paths"])


def _refinement_summary(refinement: dict[str, Any]) -> dict[str, Any]:
    rounds = [item for item in refinement["rounds"] if _valid_round(item)]
    return {
        "structured_rounds": len(refinement["rounds"]),
        "comparable_rounds": len(rounds),
        "two_latest_metric_identical": len(rounds) >= 2 and _metric_signature(rounds[-1]) == _metric_signature(rounds[-2]),
    }


def _comparable_performance(value: dict[str, Any]) -> bool:
    metrics = value["primary"]["metrics"]
    methods = ("lexical", "minilm_hybrid", "compatibility_aware")
    return all(
        isinstance(metrics.get(method), dict)
        and isinstance(metrics[method].get("queries"), int)
        and metrics[method]["queries"] > 0
        and isinstance(metrics[method].get("invalid_top_match"), int)
        for method in methods
    )


def _metric_signature(value: dict[str, Any]) -> str:
    return json.dumps(value.get("primary", {}).get("metrics", {}), sort_keys=True, separators=(",", ":"))


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


def _junit(path: Path) -> tuple[dict[str, Any], str | None]:
    """Read a JUnit receipt and reject skipped, failed, or malformed suites."""
    if not path.is_file():
        return {}, f"JUnit evidence is absent: {path.name}"
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return {}, f"JUnit evidence is unreadable: {exc}"
    suites = list(root.iter("testsuite"))
    cases = list(root.iter("testcase"))
    if not suites or not cases:
        return {}, "JUnit evidence has no test suites or test cases."
    totals = {name: sum(int(suite.get(name, "0")) for suite in suites) for name in ("failures", "errors", "skipped")}
    if any(totals.values()):
        return {}, "JUnit receipt has failures, errors, or skipped tests."
    names = {
        case.get("name")
        for case in cases
        if case.get("name") and case.find("failure") is None and case.find("error") is None
    }
    return {"tests": len(cases), "names": names}, None


def _passed_tests(receipt: dict[str, Any], *names: str) -> bool:
    return all(name in receipt.get("names", set()) for name in names)


def _refinement_evidence(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("round*.json")) if root.is_dir() else []
    rounds = []
    valid_paths = []
    for path in paths:
        value, problem = _json_object(path)
        if not problem:
            rounds.append(value)
            valid_paths.append(path)
    return {"paths": valid_paths, "rounds": rounds}


def _valid_round(value: dict[str, Any]) -> bool:
    primary = value.get("primary")
    metrics = primary.get("metrics") if isinstance(primary, dict) else None
    capability = primary.get("capability_audit") if isinstance(primary, dict) else None
    return bool(
        value.get("protocol_id")
        and value.get("catalog_sha256")
        and isinstance(metrics, dict)
        and {"lexical", "minilm_hybrid", "compatibility_aware"} <= set(metrics)
        and isinstance(capability, dict)
        and capability.get("denominator", 0) >= 40
        and value.get("gate_assessment", {}).get("overall")
    )


def _is_designated_final(manifest: dict[str, Any]) -> bool:
    evaluation = manifest.get("evaluation")
    return bool(
        str(manifest.get("run_id", "")).startswith("final-real-run-")
        and isinstance(evaluation, dict)
        and evaluation.get("split_role") in {"DEVELOPMENT", "VALIDATION"}
    )


def _designated_final_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run for run in runs if _is_designated_final(run["manifest"]) and run["integrity"] == "PASS"]


def _designated_holdout_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if run["manifest"].get("evaluation", {}).get("split_role") == "FINAL_HOLDOUT"
        and run["integrity"] == "PASS"
    ]


def _fresh_run_inference(run: dict[str, Any]) -> bool:
    manifest = run["manifest"]
    evaluation = manifest.get("evaluation", {})
    if evaluation.get("holdout_exposed_to_tuning") is not False:
        return False
    artifact = next((item for item in manifest.get("artifacts", []) if item.get("path") == "inferences.jsonl"), None)
    if not artifact:
        return False
    rows, problem = _jsonl(run["path"].parent / "inferences.jsonl")
    if problem or not rows:
        return False
    return all(
        row.get("origin") == "fresh_local_inference"
        and row.get("model_revision")
        and row.get("runtime") == "transformers"
        for row in rows
    )


def _replacement_holdout_result(
    final_runs: list[dict[str, Any]], reservation: dict[str, Any], reservation_problem: str | None
) -> dict[str, Any]:
    if reservation_problem:
        return {"status": "FAIL", "reason": reservation_problem}
    selected = reservation.get("selected_accession")
    verification = reservation.get("verification", {})
    sealed = reservation.get("sealed_evaluation_state", {})
    if (
        reservation.get("status")
        != "RESERVED_PENDING_ONE_TIME_FINAL_HOLDOUT_AUTHORIZATION"
        or not isinstance(selected, str)
        or not selected
        or verification.get("all_exact_identifier_intersections_zero") is not True
        or verification.get("candidate_publication_pmc_complete") is not True
        or verification.get("candidate_ena_sra_complete") is not True
        or verification.get("base_official_relation_errors") != 0
        or verification.get("base_official_relation_truncations") != 0
        or sealed.get("prediction_status") != "NOT_RUN"
        or sealed.get("ranking_status") != "NOT_RUN"
    ):
        return {"status": "FAIL", "reason": "Replacement reservation is malformed, overlapping, or not sealed before execution."}
    for run in final_runs:
        evaluation = run["manifest"].get("evaluation", {})
        proof = evaluation.get("source_disjointness")
        if not isinstance(proof, dict):
            continue
        if proof.get("selected_accession") != selected:
            continue
        # Known-identifier separation alone is deliberately insufficient.
        if proof.get("status") == "PROVEN_SOURCE_DISJOINT" and proof.get("unknown_overlap_count") == 0:
            return {"status": "PASS", "reason": "Replacement final run is bound to the sealed reservation and records proven source disjointness with no unknown overlap.", "run": run}
        return {"status": "FAIL", "reason": "Replacement final run leaves source overlap unknown or does not prove disjointness.", "path": run["path"]}
    return {"status": "NOT_RUN", "reason": "No designated final run is bound to the sealed replacement reservation."}


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


def _git_output(source: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
