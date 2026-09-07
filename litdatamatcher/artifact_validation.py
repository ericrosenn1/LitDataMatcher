"""Local artifact validation for completed LitDataMatcher run directories."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable

from .schemas import JsonDict


REQUIRED_ARTIFACTS: dict[str, str] = {
    "questions.jsonl": "Extracted question records.",
    "datasets.jsonl": "Candidate dataset records.",
    "matches.jsonl": "Ranked question-dataset matches.",
    "review_sheet.csv": "Flattened human review sheet.",
    "review_sheet.jsonl": "Nested review records for annotation export.",
    "source_provenance_summary.json": "Machine-readable provenance summary.",
    "module_boundary_map.json": "Advisory developer module-boundary map.",
    "provenance_transfer_check.json": "Advisory provenance transfer diagnostic.",
}

OPTIONAL_ARTIFACTS: dict[str, str] = {
    "syntheses.jsonl": "Evidence-synthesis cluster records.",
    "summary.md": "Compact run summary.",
    "publication_report.md": "Publication-oriented report with caveats.",
    "metrics.jsonl": "Run-level accounting.",
    "litdatamatcher.sqlite": "Queryable run database.",
    "evaluation.jsonl": "Optional evaluation report.",
}

REVIEW_READY_TERMS = (
    "abstract-only",
    "metadata-only",
    "curated",
    "derived capability",
    "not analysis-ready",
    "not validated",
    "not claims",
    "not computed",
    "not downloaded",
    "not patient-level",
    "hypotheses for expert review",
)


def validate_run_artifacts(run_dir: str | Path, out_dir: str | Path | None = None) -> JsonDict:
    """Inspect a completed run directory and optionally write validation outputs."""

    run_path = Path(run_dir)
    artifacts: dict[str, JsonDict] = {}
    issues: list[JsonDict] = []
    loaded: dict[str, object] = {}

    for name, role in {**REQUIRED_ARTIFACTS, **OPTIONAL_ARTIFACTS}.items():
        required = name in REQUIRED_ARTIFACTS
        status = _inspect_artifact(run_path / name, role=role, required=required)
        artifacts[name] = status
        issues.extend(status.pop("_issues", []))
        if status.get("exists") and status.get("readable"):
            loaded[name] = status.get("_payload")
        status.pop("_payload", None)

    provenance_audit = _provenance_coverage(loaded)
    advisory_audit = _advisory_artifact_audit(loaded)
    review_readiness = _review_readiness(run_path, loaded)
    sqlite_consistency = _sqlite_consistency_audit(artifacts)
    issues.extend(provenance_audit["issues"])
    issues.extend(advisory_audit["issues"])
    issues.extend(review_readiness["issues"])
    issues.extend(sqlite_consistency["issues"])

    summary = {
        "schema_version": "artifact_validation_v1",
        "run_dir": str(run_path),
        "status": _overall_status(issues),
        "artifact_counts": _artifact_counts(artifacts),
        "artifacts": artifacts,
        "provenance_audit": _without_issues(provenance_audit),
        "advisory_artifact_audit": _without_issues(advisory_audit),
        "review_readiness": _without_issues(review_readiness),
        "sqlite_consistency": _without_issues(sqlite_consistency),
        "issues": issues,
    }
    if out_dir is not None:
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "artifact_validation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "artifact_validation_report.md").write_text(
            render_artifact_validation_report(summary),
            encoding="utf-8",
        )
    return summary


def render_artifact_validation_report(summary: JsonDict) -> str:
    """Render a human-readable Markdown report for artifact review."""

    lines = [
        "# LitDataMatcher Artifact Validation Report",
        "",
        f"- Run directory: `{summary.get('run_dir', '')}`",
        f"- Overall status: **{summary.get('status', 'unknown')}**",
        f"- Artifacts present: {summary.get('artifact_counts', {}).get('present', 0)}",
        f"- Artifacts missing: {summary.get('artifact_counts', {}).get('missing', 0)}",
        "",
        "## Artifact Inventory",
        "",
        "| Artifact | Expected | Status | Records | Notes |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for name, artifact in summary.get("artifacts", {}).items():
        expected = "required" if artifact.get("required") else "optional"
        status = "ok" if artifact.get("exists") and artifact.get("readable") else "missing" if not artifact.get("exists") else "needs review"
        notes = "; ".join(str(note) for note in artifact.get("notes", []))
        lines.append(
            f"| `{name}` | {expected} | {status} | {artifact.get('records', '')} | {_md(notes)} |"
        )

    provenance = summary.get("provenance_audit", {})
    lines.extend(
        [
            "",
            "## Provenance Coverage",
            "",
            "| Surface | Records | Any provenance | Question provenance | Dataset provenance | With caveats |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for surface, counts in provenance.get("surfaces", {}).items():
        lines.append(
            f"| {surface} | {counts.get('records', 0)} | {counts.get('with_provenance', 0)} | {counts.get('question_with_provenance', 0)} | {counts.get('dataset_with_provenance', 0)} | {counts.get('with_caveats', 0)} |"
        )
    lines.extend(
        [
            "",
            f"- Source types observed: {_count_cell(provenance.get('source_types', {}))}",
            f"- Content scopes observed: {_count_cell(provenance.get('content_scopes', {}))}",
            f"- Acquisition methods observed: {_count_cell(provenance.get('acquisition_methods', {}))}",
            "",
            "## Advisory Diagnostics",
            "",
        ]
    )
    advisory = summary.get("advisory_artifact_audit", {})
    lines.extend(
        [
            f"- Module boundary map present: {advisory.get('module_boundary_map_present', False)}",
            f"- Provenance transfer check present: {advisory.get('transfer_check_present', False)}",
            f"- Transfer-check status: {advisory.get('transfer_check_status', 'unknown')}",
            f"- Treated as advisory: {advisory.get('appears_advisory', False)}",
            "",
            "## Review And Report Readiness",
            "",
        ]
    )
    readiness = summary.get("review_readiness", {})
    lines.extend(
        [
            f"- Review CSV rows: {readiness.get('review_csv_rows', 0)}",
            f"- Review JSONL rows: {readiness.get('review_jsonl_rows', 0)}",
            f"- Publication report present: {readiness.get('publication_report_present', False)}",
            f"- Conservative caveat terms found: {_count_cell(readiness.get('caveat_terms_found', {}))}",
            "",
            "## SQLite Consistency",
            "",
        ]
    )
    sqlite_consistency = summary.get("sqlite_consistency", {})
    lines.extend(
        [
            f"- SQLite present: {sqlite_consistency.get('sqlite_present', False)}",
            f"- Counts match JSONL artifacts: {sqlite_consistency.get('counts_match', False)}",
            "| Table | SQLite rows | JSONL records | Status |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for table, counts in sqlite_consistency.get("tables", {}).items():
        status = "ok" if counts.get("matches") else "mismatch"
        lines.append(
            f"| {table} | {counts.get('sqlite_records', '')} | {counts.get('artifact_records', '')} | {status} |"
        )
    lines.extend(
        [
            "",
            "## Issues",
            "",
        ]
    )
    issues = list(summary.get("issues", []))
    if not issues:
        lines.append("- No issues detected by this advisory validator.")
    else:
        for issue in issues:
            lines.append(
                f"- **{issue.get('severity', 'info')}** `{issue.get('category', 'general')}`: {_md(issue.get('message', ''))}"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This report checks artifact completeness, readability, provenance visibility, and review caveats. It does not validate scientific correctness, answerability, source licenses, live database coverage, or whether a dataset is analysis-ready.",
            "",
        ]
    )
    return "\n".join(lines)


def _inspect_artifact(path: Path, *, role: str, required: bool) -> JsonDict:
    """Read one artifact without raising on malformed content."""

    result: JsonDict = {
        "path": str(path),
        "role": role,
        "required": required,
        "exists": path.exists(),
        "readable": False,
        "bytes": 0,
        "records": 0,
        "format": _format_for_path(path),
        "notes": [],
        "_issues": [],
        "_payload": None,
    }
    if not path.exists():
        severity = "error" if required else "info"
        result["_issues"].append(_issue(severity, "missing_artifact", f"{path.name} is missing."))
        return result
    result["bytes"] = path.stat().st_size
    if path.stat().st_size == 0:
        result["_issues"].append(_issue("error" if required else "info", "empty_artifact", f"{path.name} is empty."))
        return result
    try:
        if path.suffix == ".jsonl":
            payload, errors = _read_jsonl_lenient(path)
            result["records"] = len(payload)
            result["_payload"] = payload
            result["readable"] = not errors
            for error in errors:
                result["_issues"].append(_issue("error", "malformed_jsonl", error))
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            result["_payload"] = payload
            result["records"] = len(payload) if isinstance(payload, list) else 1
            result["readable"] = True
        elif path.suffix == ".csv":
            rows, errors = _read_csv_lenient(path)
            result["records"] = len(rows)
            result["_payload"] = rows
            result["readable"] = not errors
            for error in errors:
                result["_issues"].append(_issue("error", "malformed_csv", error))
        elif path.suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            result["_payload"] = text
            result["records"] = len([line for line in text.splitlines() if line.strip()])
            result["readable"] = True
        elif path.suffix == ".sqlite":
            result.update(_inspect_sqlite(path))
            result["readable"] = bool(result.get("sqlite_readable"))
        else:
            result["readable"] = True
            result["notes"].append("Binary or unparsed artifact; existence checked only.")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, sqlite3.Error) as exc:
        result["_issues"].append(_issue("error", "unreadable_artifact", f"{path.name} could not be read: {exc}"))
    if result["readable"] and required and result["records"] == 0:
        result["_issues"].append(_issue("warning", "empty_required_records", f"{path.name} has no records."))
    return result


def _provenance_coverage(loaded: dict[str, object]) -> JsonDict:
    """Audit provenance visibility across run surfaces."""

    issues: list[JsonDict] = []
    surfaces = {
        "questions": _surface_counts(loaded.get("questions.jsonl", []), side="question"),
        "datasets": _surface_counts(loaded.get("datasets.jsonl", []), side="dataset"),
        "matches": _match_surface_counts(loaded.get("matches.jsonl", [])),
        "review_jsonl": _review_surface_counts(loaded.get("review_sheet.jsonl", [])),
        "review_csv": _review_surface_counts(loaded.get("review_sheet.csv", [])),
    }
    source_types: Counter[str] = Counter()
    content_scopes: Counter[str] = Counter()
    acquisition_methods: Counter[str] = Counter()
    for rows in (
        loaded.get("questions.jsonl", []),
        loaded.get("datasets.jsonl", []),
        loaded.get("matches.jsonl", []),
        loaded.get("review_sheet.jsonl", []),
        loaded.get("review_sheet.csv", []),
    ):
        for provenance in _iter_provenance(rows):
            source_types[str(provenance.get("source_type", "unknown") or "unknown")] += 1
            content_scopes[str(provenance.get("content_scope", "unknown") or "unknown")] += 1
            acquisition_methods[str(provenance.get("acquisition_method", "unknown") or "unknown")] += 1

    for surface, counts in surfaces.items():
        if counts["records"] and not counts["with_provenance"]:
            issues.append(_issue("warning", "missing_provenance", f"{surface} records do not expose source provenance."))
        if counts.get("malformed_provenance"):
            issues.append(
                _issue(
                    "warning",
                    "malformed_provenance",
                    f"{surface} has {counts['malformed_provenance']} malformed provenance value(s).",
                )
            )
        if surface in {"review_csv", "review_jsonl"} and counts["records"]:
            if not counts.get("dataset_with_provenance"):
                issues.append(
                    _issue(
                        "warning",
                        "review_dataset_provenance_hidden",
                        f"{surface} does not expose populated dataset-side provenance.",
                    )
                )
    if "source_provenance_summary.json" in loaded and isinstance(
        loaded.get("source_provenance_summary.json"), dict
    ):
        summary = loaded["source_provenance_summary.json"]
        if not summary.get("source_types"):
            issues.append(_issue("warning", "summary_missing_source_types", "source_provenance_summary.json has no source type counts."))
        if not summary.get("review_caveats"):
            issues.append(_issue("warning", "summary_missing_caveats", "source_provenance_summary.json has no review caveat counts."))
    elif "source_provenance_summary.json" in loaded:
        issues.append(_issue("warning", "summary_unreadable", "source_provenance_summary.json was not loaded as an object."))

    return {
        "surfaces": surfaces,
        "source_types": dict(sorted(source_types.items())),
        "content_scopes": dict(sorted(content_scopes.items())),
        "acquisition_methods": dict(sorted(acquisition_methods.items())),
        "curated_catalog_provenance_seen": "curated_biomedical_catalog" in source_types,
        "issues": issues,
    }


def _advisory_artifact_audit(loaded: dict[str, object]) -> JsonDict:
    """Check diagnostic artifacts without treating them as hard validation."""

    issues: list[JsonDict] = []
    boundary = loaded.get("module_boundary_map.json")
    transfer = loaded.get("provenance_transfer_check.json")
    boundary_present = isinstance(boundary, dict)
    transfer_present = isinstance(transfer, dict)
    if boundary_present and not any("provenance" in key for key in boundary):
        issues.append(_issue("warning", "boundary_map_thin", "module_boundary_map.json does not include a provenance module boundary."))
    if transfer_present:
        status = str(transfer.get("status", "unknown"))
        if status not in {"pass", "needs_review"}:
            issues.append(_issue("warning", "transfer_status_unknown", f"Unexpected provenance transfer status: {status}."))
    else:
        status = "missing"
    return {
        "module_boundary_map_present": boundary_present,
        "module_boundary_count": len(boundary) if isinstance(boundary, dict) else 0,
        "transfer_check_present": transfer_present,
        "transfer_check_status": status,
        "transfer_issue_count": len(transfer.get("issues", [])) if isinstance(transfer, dict) else 0,
        "appears_advisory": transfer_present and status in {"pass", "needs_review"},
        "issues": issues,
    }


def _review_readiness(run_path: Path, loaded: dict[str, object]) -> JsonDict:
    """Inspect review/export/report surfaces for conservative caveat language."""

    issues: list[JsonDict] = []
    review_csv = loaded.get("review_sheet.csv", [])
    review_jsonl = loaded.get("review_sheet.jsonl", [])
    report_text = str(loaded.get("publication_report.md", "") or "")
    caveat_terms = _term_counts(report_text)
    if isinstance(review_csv, list) and review_csv:
        if not any(_record_caveats(row) for row in review_csv if isinstance(row, dict)):
            issues.append(_issue("warning", "review_csv_no_caveats", "review_sheet.csv has rows but no source caveat values."))
    if isinstance(review_jsonl, list) and review_jsonl:
        if not any(_record_caveats(row) for row in review_jsonl if isinstance(row, dict)):
            issues.append(_issue("warning", "review_jsonl_no_caveats", "review_sheet.jsonl has rows but no source caveat values."))
    if report_text and not any(caveat_terms.values()):
        issues.append(_issue("warning", "report_caveats_not_detected", "publication_report.md lacks expected conservative caveat terms."))
    annotation_dir = run_path / "annotations"
    annotation_outputs = sorted(path.name for path in annotation_dir.glob("*") if annotation_dir.exists())
    return {
        "review_csv_rows": len(review_csv) if isinstance(review_csv, list) else 0,
        "review_jsonl_rows": len(review_jsonl) if isinstance(review_jsonl, list) else 0,
        "publication_report_present": bool(report_text),
        "caveat_terms_found": {key: value for key, value in caveat_terms.items() if value},
        "annotation_outputs_present": annotation_outputs,
        "issues": issues,
    }


def _sqlite_consistency_audit(artifacts: dict[str, JsonDict]) -> JsonDict:
    """Compare SQLite per-run tables with canonical JSONL artifact counts."""

    issues: list[JsonDict] = []
    sqlite_artifact = artifacts.get("litdatamatcher.sqlite", {})
    sqlite_counts = sqlite_artifact.get("sqlite_counts", {})
    table_to_artifact = {
        "questions": "questions.jsonl",
        "datasets": "datasets.jsonl",
        "syntheses": "syntheses.jsonl",
        "matches": "matches.jsonl",
    }
    tables: dict[str, JsonDict] = {}
    if not sqlite_artifact.get("exists"):
        return {
            "sqlite_present": False,
            "counts_match": False,
            "tables": tables,
            "issues": issues,
        }
    for table, artifact_name in table_to_artifact.items():
        artifact = artifacts.get(artifact_name, {})
        if not artifact.get("exists") or not artifact.get("readable"):
            continue
        sqlite_records = int(sqlite_counts.get(table, -1)) if isinstance(sqlite_counts, dict) else -1
        artifact_records = int(artifact.get("records", 0))
        matches = sqlite_records == artifact_records
        tables[table] = {
            "sqlite_records": sqlite_records,
            "artifact_records": artifact_records,
            "matches": matches,
            "artifact": artifact_name,
        }
        if not matches:
            issues.append(
                _issue(
                    "warning",
                    "sqlite_artifact_count_mismatch",
                    (
                        f"litdatamatcher.sqlite table {table} has {sqlite_records} rows, "
                        f"but {artifact_name} has {artifact_records} records."
                    ),
                )
            )
    return {
        "sqlite_present": True,
        "counts_match": all(row.get("matches") for row in tables.values()) if tables else False,
        "tables": tables,
        "interpretation": (
            "For a fresh pipeline run, SQLite per-run tables should match the canonical "
            "JSONL artifact record counts for questions, datasets, syntheses, and matches."
        ),
        "issues": issues,
    }


def _surface_counts(rows: object, *, side: str) -> JsonDict:
    """Count provenance and caveats on top-level question or dataset records."""

    records = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    with_provenance = 0
    with_caveats = 0
    states: Counter[str] = Counter()
    for row in records:
        entries, state = _standard_provenance(row)
        states[state] += 1
        if entries:
            with_provenance += 1
        if _record_caveats(row):
            with_caveats += 1
    question_with = with_provenance if side == "question" else 0
    dataset_with = with_provenance if side == "dataset" else 0
    return _coverage_counts(
        records=len(records),
        with_provenance=with_provenance,
        with_caveats=with_caveats,
        question_with_provenance=question_with,
        dataset_with_provenance=dataset_with,
        states=states,
    )


def _match_surface_counts(rows: object) -> JsonDict:
    """Count provenance visible through nested match question/dataset records."""

    records = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    with_provenance = 0
    question_with_provenance = 0
    dataset_with_provenance = 0
    with_caveats = 0
    states: Counter[str] = Counter()
    for row in records:
        question = row.get("question", {})
        dataset = row.get("dataset", {})
        question_entries, question_state = _standard_provenance(
            question if isinstance(question, dict) else {}
        )
        dataset_entries, dataset_state = _standard_provenance(
            dataset if isinstance(dataset, dict) else {}
        )
        states[_combine_states([question_state, dataset_state])] += 1
        if question_entries:
            question_with_provenance += 1
        if dataset_entries:
            dataset_with_provenance += 1
        if question_entries or dataset_entries:
            with_provenance += 1
        if _record_caveats(row):
            with_caveats += 1
    return _coverage_counts(
        records=len(records),
        with_provenance=with_provenance,
        with_caveats=with_caveats,
        question_with_provenance=question_with_provenance,
        dataset_with_provenance=dataset_with_provenance,
        states=states,
    )


def _review_surface_counts(rows: object) -> JsonDict:
    """Count question- and dataset-side provenance in review exports."""

    records = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    with_provenance = 0
    question_with_provenance = 0
    dataset_with_provenance = 0
    with_caveats = 0
    states: Counter[str] = Counter()
    for row in records:
        question_entries, question_state = _review_side_provenance(row, "question")
        dataset_entries, dataset_state = _review_side_provenance(row, "dataset")
        states[_combine_states([question_state, dataset_state])] += 1
        if question_entries:
            question_with_provenance += 1
        if dataset_entries:
            dataset_with_provenance += 1
        if question_entries or dataset_entries:
            with_provenance += 1
        if _record_caveats(row):
            with_caveats += 1
    return _coverage_counts(
        records=len(records),
        with_provenance=with_provenance,
        with_caveats=with_caveats,
        question_with_provenance=question_with_provenance,
        dataset_with_provenance=dataset_with_provenance,
        states=states,
    )


def _iter_provenance(rows: object) -> Iterable[JsonDict]:
    """Yield provenance dictionaries from records and nested match objects."""

    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        for item in _standard_provenance(row)[0]:
            yield item
        for field in ("question", "dataset"):
            nested = row.get(field, {})
            if isinstance(nested, dict):
                for item in _standard_provenance(nested)[0]:
                    yield item
        for side in ("question", "dataset"):
            for item in _review_side_provenance(row, side)[0]:
                yield item


def _record_caveats(row: JsonDict) -> list[str]:
    """Return caveats visible on a record or side-specific review fields."""

    caveats: list[str] = []
    for field in ("source_caveats", "question_source_caveats", "dataset_source_caveats"):
        values = row.get(field, [])
        if isinstance(values, str):
            caveats.extend(part.strip() for part in values.split(";") if part.strip())
        elif isinstance(values, list):
            caveats.extend(str(value).strip() for value in values if str(value).strip())
    return _dedupe_strings(caveats)


def _coverage_counts(
    *,
    records: int,
    with_provenance: int,
    with_caveats: int,
    question_with_provenance: int,
    dataset_with_provenance: int,
    states: Counter[str],
) -> JsonDict:
    """Build a stable coverage object for reports and tests."""

    state_counts = {state: states.get(state, 0) for state in ("populated", "empty", "malformed", "absent")}
    return {
        "records": records,
        "with_provenance": with_provenance,
        "question_with_provenance": question_with_provenance,
        "dataset_with_provenance": dataset_with_provenance,
        "with_caveats": with_caveats,
        "provenance_states": state_counts,
        "empty_provenance": state_counts["empty"],
        "malformed_provenance": state_counts["malformed"],
        "absent_provenance": state_counts["absent"],
    }


def _standard_provenance(record: JsonDict) -> tuple[list[JsonDict], str]:
    """Return populated provenance and state for normal run records."""

    values: list[object] = []
    if "source_provenance" in record:
        values.append(record.get("source_provenance"))
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        if "source_provenance" in metadata:
            values.append(metadata.get("source_provenance"))
        if "source_provenances" in metadata:
            values.append(metadata.get("source_provenances"))
    entries, state = _entries_from_values(values)
    if entries:
        return entries, "populated"
    return entries, state


def _review_side_provenance(row: JsonDict, side: str) -> tuple[list[JsonDict], str]:
    """Return populated provenance and state for a review row side."""

    values: list[object] = []
    json_values: list[object] = []
    flat_fields: tuple[str, ...]
    if side == "question":
        values.extend(
            row.get(field)
            for field in ("question_source_provenance", "source_provenance")
            if field in row
        )
        json_values.extend(
            row.get(field)
            for field in ("question_source_provenance_json", "source_provenance_json")
            if field in row
        )
        flat_fields = (
            "question_source_types",
            "question_source_content_scopes",
            "question_source_acquisition_methods",
            "source_types",
            "source_content_scopes",
            "source_acquisition_methods",
        )
        match = row.get("match", {})
        nested = match.get("question", {}) if isinstance(match, dict) else {}
    else:
        values.extend(
            row.get(field)
            for field in ("dataset_source_provenance",)
            if field in row
        )
        json_values.extend(
            row.get(field)
            for field in ("dataset_source_provenance_json",)
            if field in row
        )
        flat_fields = (
            "dataset_source_types",
            "dataset_source_content_scopes",
            "dataset_source_acquisition_methods",
        )
        match = row.get("match", {})
        nested = match.get("dataset", {}) if isinstance(match, dict) else {}

    entries, state = _entries_from_values(values)
    json_entries, json_state = _entries_from_values(json_values, parse_json=True)
    nested_entries, nested_state = _standard_provenance(nested if isinstance(nested, dict) else {})
    entries = _dedupe_provenance([*entries, *json_entries, *nested_entries])
    if entries:
        return entries, "populated"
    if any(str(row.get(field, "") or "").strip() for field in flat_fields):
        return [], "populated"
    return [], _combine_states([state, json_state, nested_state])


def _entries_from_values(values: Iterable[object], *, parse_json: bool = False) -> tuple[list[JsonDict], str]:
    """Extract provenance dictionaries and classify empty/malformed containers."""

    entries: list[JsonDict] = []
    states: list[str] = []
    for value in values:
        parsed = value
        if parse_json:
            text = str(value or "").strip()
            if not text:
                states.append("absent")
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                states.append("malformed")
                continue
        item_entries, state = _entries_from_value(parsed)
        entries.extend(item_entries)
        states.append(state)
    entries = _dedupe_provenance(entries)
    if entries:
        return entries, "populated"
    return [], _combine_states(states)


def _entries_from_value(value: object) -> tuple[list[JsonDict], str]:
    """Classify one provenance value without treating empty containers as present."""

    if value is None:
        return [], "absent"
    if isinstance(value, str):
        return ([], "absent") if not value.strip() else ([], "malformed")
    if isinstance(value, dict):
        return ([value], "populated") if value else ([], "empty")
    if isinstance(value, list):
        if not value:
            return [], "empty"
        entries = [item for item in value if isinstance(item, dict) and item]
        malformed = any(not isinstance(item, dict) for item in value)
        if entries:
            return entries, "populated"
        return [], "malformed" if malformed else "empty"
    return [], "malformed"


def _combine_states(states: Iterable[str]) -> str:
    """Collapse per-field provenance states into one conservative state."""

    states = [state for state in states if state]
    for state in ("populated", "malformed", "empty"):
        if state in states:
            return state
    return "absent"


def _dedupe_provenance(values: list[JsonDict]) -> list[JsonDict]:
    """De-duplicate provenance records by locator, type, and content scope."""

    out: list[JsonDict] = []
    seen: set[tuple[str, str, str]] = set()
    for item in values:
        key = (
            str(item.get("source_locator", "") or item.get("source_url", "") or item.get("local_path", "")),
            str(item.get("source_type", "")),
            str(item.get("content_scope", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _dedupe_strings(values: Iterable[object]) -> list[str]:
    """Return nonblank strings with order-preserving de-duplication."""

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _read_jsonl_lenient(path: Path) -> tuple[list[JsonDict], list[str]]:
    """Read JSONL rows while collecting malformed-line errors."""

    rows: list[JsonDict] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_number}: {exc}")
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
            else:
                errors.append(f"{path.name}:{line_number}: row is {type(parsed).__name__}, expected object")
    return rows, errors


def _read_csv_lenient(path: Path) -> tuple[list[JsonDict], list[str]]:
    """Read CSV rows while collecting parse-level errors."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
            if not reader.fieldnames:
                return rows, [f"{path.name}: missing header row"]
            return rows, []
    except csv.Error as exc:
        return [], [f"{path.name}: {exc}"]


def _inspect_sqlite(path: Path) -> JsonDict:
    """Return SQLite table counts for known run tables."""

    counts: dict[str, int] = {}
    conn = sqlite3.connect(path)
    try:
        for table in ("questions", "datasets", "syntheses", "matches"):
            try:
                counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            except sqlite3.Error:
                counts[table] = -1
    finally:
        conn.close()
    return {"sqlite_readable": True, "sqlite_counts": counts, "records": sum(max(0, value) for value in counts.values())}


def _format_for_path(path: Path) -> str:
    """Return a simple artifact format name from path suffix."""

    if path.suffix == ".jsonl":
        return "jsonl"
    if path.suffix == ".json":
        return "json"
    if path.suffix == ".csv":
        return "csv"
    if path.suffix == ".md":
        return "markdown"
    if path.suffix == ".sqlite":
        return "sqlite"
    return path.suffix.lstrip(".") or "unknown"


def _artifact_counts(artifacts: dict[str, JsonDict]) -> JsonDict:
    """Summarize artifact presence and readability."""

    return {
        "expected": len(artifacts),
        "present": sum(1 for item in artifacts.values() if item.get("exists")),
        "missing": sum(1 for item in artifacts.values() if not item.get("exists")),
        "readable": sum(1 for item in artifacts.values() if item.get("readable")),
        "required_missing": sum(1 for item in artifacts.values() if item.get("required") and not item.get("exists")),
    }


def _overall_status(issues: list[JsonDict]) -> str:
    """Return a compact status from issue severities."""

    severities = {issue.get("severity") for issue in issues}
    if "error" in severities:
        return "fail"
    if "warning" in severities:
        return "needs_review"
    return "pass"


def _without_issues(payload: JsonDict) -> JsonDict:
    """Return a copy without embedded issues for top-level summary nesting."""

    result = dict(payload)
    result.pop("issues", None)
    return result


def _term_counts(text: str) -> JsonDict:
    """Count conservative caveat terms in a report body."""

    lowered = text.lower()
    return {term: lowered.count(term) for term in REVIEW_READY_TERMS}


def _count_cell(values: object) -> str:
    """Render compact count dictionaries for Markdown."""

    if not isinstance(values, dict) or not values:
        return "none"
    return "; ".join(f"{key}: {value}" for key, value in sorted(values.items()))


def _issue(severity: str, category: str, message: str) -> JsonDict:
    """Build a stable issue record."""

    return {"severity": severity, "category": category, "message": message}


def _md(value: object) -> str:
    """Escape generated Markdown table text."""

    return str(value or "").replace("|", "/")
