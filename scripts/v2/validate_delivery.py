"""Validate the immutable LitDataMatcher v2 delivery and emit closeout evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_integrity(run: Path) -> bool:
    manifest = _json(run / "RUN_MANIFEST.json")
    if manifest.get("execution_status") != "PASS" or manifest.get("failures"):
        return False
    artifacts = {item.get("path"): item for item in manifest.get("artifacts", [])}
    for name, item in artifacts.items():
        path = (run / str(name)).resolve()
        if path.parent != run.resolve() or not path.is_file():
            return False
        if (
            item.get("validation") != "PASS"
            or item.get("size_bytes") != path.stat().st_size
            or item.get("sha256") != _sha256(path)
        ):
            return False
    return bool(artifacts) and all(
        command.get("exit_code") == 0 and command.get("log_reference") in artifacts
        for command in manifest.get("commands", [])
    )


def _traceable_claims(run: Path) -> bool:
    rows = [
        json.loads(line)
        for line in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return bool(rows) and all(
        row.get("source_document_id")
        and row.get("source_locator")
        and isinstance(row.get("evidence_span"), dict)
        and row["evidence_span"].get("text") == row.get("statement")
        for row in rows
    )


def _archives_pass(release: Path, release_manifest: dict) -> bool:
    entries = release_manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        return False
    for item in entries:
        path = (release / item.get("path", "")).resolve()
        if path.parent != release.resolve() or not path.is_file():
            return False
        if item.get("sha256") != _sha256(path) or item.get("size_bytes") != path.stat().st_size:
            return False
        if path.suffix == ".whl" or path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    return False
        elif path.name.endswith(".tar.gz"):
            with tarfile.open(path, "r:gz") as archive:
                if not archive.getmembers():
                    return False
    return True


def validate(args) -> dict:
    source = args.source.resolve()
    data = args.data.resolve()
    release = args.release.resolve()
    final_run = args.final_run.resolve()
    holdout_run = args.holdout_run.resolve()
    cleanroom = _json(args.cleanroom)
    independent = _json(args.independent_review)
    worker = _json(args.worker_evidence)
    state = _json(source / "project_state" / "TASK_STATE.json")
    supervisor = _json(data / "operations" / "supervisor_activation.json")
    release_manifest = _json(release / "RELEASE_MANIFEST.json")
    holdout_manifest = _json(holdout_run / "RUN_MANIFEST.json")
    remote = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "refs/remotes/origin/codex/litdatamatcher-v2-build"],
        text=True,
    ).strip()
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    report_text = (final_run / "report.html").read_text(encoding="utf-8")
    with (final_run / "review_sheet.csv").open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    archive_pass = _archives_pass(release, release_manifest)
    final_pass = _manifest_integrity(final_run)
    holdout_pass = (
        _manifest_integrity(holdout_run)
        and holdout_manifest.get("evaluation", {}).get("split_role") == "FINAL_HOLDOUT"
        and holdout_manifest.get("evaluation", {})
        .get("source_disjointness", {})
        .get("unknown_overlap_count")
        == 0
    )
    lock_notice = all(
        (source / path).is_file()
        for path in (
            "litdatamatcher/schemas_v2/PACKAGE_MANIFEST.json",
            "litdatamatcher/schemas_v2/requirements-v2.lock",
            "docs/v2/THIRD_PARTY_NOTICES.md",
        )
    )
    checks = {
        "archive_reopen_pass": archive_pass,
        "source_archive_integrity_pass": archive_pass and remote == head,
        "version_lock_manifest_notice_pass": lock_notice,
        "independent_readiness_agreement": independent.get("status") == "PASS",
        "status_stop_reason_agreement": state.get("execution_status") == "COMPLETE"
        and state.get("product_status") == "HARDENED_ALPHA_READY"
        and bool(state.get("stop_reason")),
        "stored_output_report_pass": final_pass and "Content-Security-Policy" in report_text,
        "traceable_claims_pass": final_pass and _traceable_claims(final_run),
        "clean_install_pass": cleanroom.get("clean_install_pass") is True,
        "new_document_input_pass": cleanroom.get("new_document_input_pass") is True,
        "topic_input_pass": cleanroom.get("topic_input_pass") is True,
        "explicit_question_input_pass": cleanroom.get("explicit_question_input_pass") is True,
        "opportunity_review_pass": final_pass and bool(review_rows),
        "score_explanation_pass": final_pass and "uncalibrated heuristic" in report_text,
        "independent_review_pass": independent.get("status") == "PASS",
        "worker_passes_minimum": worker.get("minimum_substantive_passes", 0),
        "integrated_refinement_rounds": worker.get("integrated_refinement_rounds", 0),
        "overlapping_jobs_pass": _json(
            data / "evaluation" / "concurrency" / "final" / "concurrency_validation.json"
        ).get("status")
        == "PASS",
        "completion_supervisor_disabled_or_idle": supervisor.get("status")
        == "DISABLED_AT_COMPLETION",
        "delivery_owner_stop_conditions": state.get("execution_status") == "COMPLETE"
        and not state.get("blockers"),
        "final_integrated_run_pass": final_pass,
        "final_holdout_pass": holdout_pass,
    }
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    result = {
        "schema_version": "delivery-validation-v1",
        "status": "PASS"
        if all(boolean_checks)
        and checks["worker_passes_minimum"] >= 2
        and checks["integrated_refinement_rounds"] >= 3
        else "FAIL",
        "source_commit": head,
        "release_manifest_sha256": _sha256(release / "RELEASE_MANIFEST.json"),
        "final_run_manifest_sha256": _sha256(final_run / "RUN_MANIFEST.json"),
        "holdout_run_manifest_sha256": _sha256(holdout_run / "RUN_MANIFEST.json"),
        **checks,
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--cleanroom", type=Path, required=True)
    parser.add_argument("--final-run", type=Path, required=True)
    parser.add_argument("--holdout-run", type=Path, required=True)
    parser.add_argument("--independent-review", type=Path, required=True)
    parser.add_argument("--worker-evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
