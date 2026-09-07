"""Human review export and label-ingestion helpers.

CSV exports are optimized for human scoring, while JSONL exports preserve
nested match/provenance context for later annotation and training workflows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .provenance import provenance_from_record, provenance_review_caveats
from .schemas import MatchCandidate, QuestionDataMatchLabel, QuestionQualityScore
from .storage import read_jsonl, write_jsonl


REVIEW_FIELDNAMES = [
    "rank",
    "match_id",
    "question_id",
    "dataset_id",
    "score",
    "score_variable_overlap",
    "score_semantic_relevance",
    "score_population_fit",
    "score_data_quality",
    "score_sample_adequacy",
    "score_significance",
    "score_feasibility",
    "score_governance",
    "score_design_fit",
    "score_uncertainty_penalty",
    "question",
    "primary_source_id",
    "source_ids",
    "document_id",
    "document_ids",
    "evidence_dois",
    "evidence_titles",
    "evidence_sections",
    "evidence_sentence_indices",
    "source_types",
    "source_content_scopes",
    "source_acquisition_methods",
    "source_provenance_statuses",
    "source_limitations",
    "source_warnings",
    "source_caveats",
    "source_provenance_json",
    "question_source_types",
    "question_source_content_scopes",
    "question_source_acquisition_methods",
    "question_source_provenance_statuses",
    "question_source_limitations",
    "question_source_warnings",
    "question_source_caveats",
    "question_source_provenance_json",
    "dataset_title",
    "dataset_source",
    "dataset_source_types",
    "dataset_source_content_scopes",
    "dataset_source_acquisition_methods",
    "dataset_source_provenance_statuses",
    "dataset_source_limitations",
    "dataset_source_warnings",
    "dataset_source_caveats",
    "dataset_source_provenance_json",
    "dataset_capability_categories",
    "dataset_capability_summary",
    "direct_capability_support",
    "proxy_capability_support",
    "missing_capability_support",
    "match_answerability_class",
    "dataset_capability_caveats",
    "required_variables",
    "missing_variables",
    "rationale",
    "recommended_design",
    "feasibility_overall",
    "feasibility_variable_coverage",
    "feasibility_population_fit",
    "feasibility_sample_adequacy",
    "feasibility_longitudinal_fit",
    "feasibility_assay_fit",
    "governance_reuse_score",
    "governance_access_score",
    "governance_license_score",
    "governance_privacy_score",
    "governance_risk_flags",
    "score_components_json",
    "assessments_json",
    "match_relevance",
    "expert_question_quality",
    "expert_data_match_quality",
    "expert_notes",
]


def _json_cell(value: object) -> str:
    """Serialize nested review context into a stable CSV cell."""

    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _clean_review_list(values: Iterable[object]) -> list[str]:
    """Return stable nonblank strings for compact provenance fields."""

    cleaned: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _list_cell(values: Iterable[object]) -> str:
    """Format short provenance lists for CSV review sheets."""

    return "; ".join(_clean_review_list(values))


def _provenance_metadata(provenance: Iterable[dict]) -> dict[str, object]:
    """Return compact, reviewer-facing fields for one provenance side."""

    provenance = [item for item in provenance if isinstance(item, dict) and item]
    return {
        "source_provenance": provenance,
        "source_types": _clean_review_list(item.get("source_type", "") for item in provenance),
        "source_content_scopes": _clean_review_list(
            item.get("content_scope", "") for item in provenance
        ),
        "source_acquisition_methods": _clean_review_list(
            item.get("acquisition_method", "") for item in provenance
        ),
        "source_provenance_statuses": _clean_review_list(
            item.get("status", "") for item in provenance
        ),
        "source_limitations": _clean_review_list(
            limitation
            for item in provenance
            for limitation in item.get("limitations", []) or []
        ),
        "source_warnings": _clean_review_list(
            warning for item in provenance for warning in item.get("warnings", []) or []
        ),
        "source_caveats": provenance_review_caveats(provenance),
    }


def _question_source_metadata(match: MatchCandidate) -> dict[str, object]:
    """Collect source/document metadata from a match's question evidence."""

    question = match.question
    evidence = list(question.evidence)
    source_ids = _clean_review_list(
        [*question.source_ids, *(item.source_id for item in evidence)]
    )
    primary_source_id = source_ids[0] if source_ids else ""
    metadata = dict(question.metadata or {})
    source_metadata = _provenance_metadata(_question_provenance_records(metadata))
    # These fields provide review context; they do not prove the match is correct.
    document_ids = _clean_review_list(
        [
            metadata.get("document_id", ""),
            *metadata.get("document_ids", []),
        ]
        if isinstance(metadata.get("document_ids", []), list)
        else [metadata.get("document_id", ""), metadata.get("document_ids", "")]
    )
    return {
        "primary_source_id": primary_source_id,
        "source_ids": source_ids,
        "document_id": document_ids[0] if document_ids else "",
        "document_ids": document_ids,
        "evidence_dois": _clean_review_list(item.doi for item in evidence),
        "evidence_titles": _clean_review_list(item.title for item in evidence),
        "evidence_sections": _clean_review_list(item.section for item in evidence),
        "evidence_sentence_indices": _clean_review_list(
            item.sentence_index for item in evidence if item.sentence_index >= 0
        ),
        **source_metadata,
    }


def _dataset_source_metadata(match: MatchCandidate) -> dict[str, object]:
    """Collect source/catalog provenance from a match's dataset record."""

    return _provenance_metadata(provenance_from_record(match.dataset.to_dict()))


def _capability_support_metadata(match: MatchCandidate) -> dict[str, object]:
    """Return review-facing capability support fields for one match."""

    support = match.assessments.get("capability_support", {})
    if not isinstance(support, dict):
        support = {}
    annotations = []
    metadata = match.dataset.metadata if isinstance(match.dataset.metadata, dict) else {}
    raw_annotations = metadata.get("capability_annotations", [])
    if isinstance(raw_annotations, list):
        annotations = [item for item in raw_annotations if isinstance(item, dict)]
    summary = []
    for item in annotations:
        capability = str(item.get("capability", "") or "").strip()
        support_level = str(item.get("support", "") or "").strip()
        capability_type = str(item.get("capability_type", "") or "").strip()
        if capability:
            summary.append(
                ":".join(part for part in (capability, capability_type, support_level) if part)
            )
    return {
        "dataset_capability_categories": support.get("dataset_capability_categories", []),
        "dataset_capability_summary": summary,
        "direct_capability_support": support.get("direct_capabilities", []),
        "proxy_capability_support": support.get("proxy_capabilities", []),
        "missing_capability_support": support.get("missing_capabilities", []),
        "match_answerability_class": support.get("answerability_class", ""),
        "dataset_capability_caveats": support.get("capability_caveats", []),
    }


def _question_provenance_records(metadata: dict) -> list[dict]:
    """Return provenance records stored on a question metadata object."""

    provenance = metadata.get("source_provenance", {})
    if isinstance(provenance, dict) and provenance:
        return [provenance]
    if isinstance(provenance, list):
        return [item for item in provenance if isinstance(item, dict) and item]
    return []


def match_review_rows(matches: Iterable[MatchCandidate]) -> list[dict]:
    """Flatten ranked matches into rows suitable for reviewer scoring."""

    rows: list[dict] = []
    for rank, match in enumerate(matches, 1):
        score = match.score.to_dict()
        feasibility = match.assessments.get("feasibility", {})
        governance = match.assessments.get("governance", {})
        risk_flags = governance.get("risk_flags", [])
        source_metadata = _question_source_metadata(match)
        dataset_source_metadata = _dataset_source_metadata(match)
        capability_metadata = _capability_support_metadata(match)
        # Keep stable IDs beside readable text so expert labels can map back to objects.
        rows.append(
            {
                "rank": rank,
                "match_id": match.match_id,
                "question_id": match.question.question_id,
                "dataset_id": match.dataset.dataset_id,
                "score": match.score.combined,
                "score_variable_overlap": score["variable_overlap"],
                "score_semantic_relevance": score["semantic_relevance"],
                "score_population_fit": score["population_fit"],
                "score_data_quality": score["data_quality"],
                "score_sample_adequacy": score["sample_adequacy"],
                "score_significance": score["significance"],
                "score_feasibility": score["feasibility"],
                "score_governance": score["governance"],
                "score_design_fit": score["design_fit"],
                "score_uncertainty_penalty": score["uncertainty_penalty"],
                "question": match.question.question,
                "primary_source_id": source_metadata["primary_source_id"],
                "source_ids": _list_cell(source_metadata["source_ids"]),
                "document_id": source_metadata["document_id"],
                "document_ids": _list_cell(source_metadata["document_ids"]),
                "evidence_dois": _list_cell(source_metadata["evidence_dois"]),
                "evidence_titles": _list_cell(source_metadata["evidence_titles"]),
                "evidence_sections": _list_cell(source_metadata["evidence_sections"]),
                "evidence_sentence_indices": _list_cell(
                    source_metadata["evidence_sentence_indices"]
                ),
                "source_types": _list_cell(source_metadata["source_types"]),
                "source_content_scopes": _list_cell(source_metadata["source_content_scopes"]),
                "source_acquisition_methods": _list_cell(
                    source_metadata["source_acquisition_methods"]
                ),
                "source_provenance_statuses": _list_cell(
                    source_metadata["source_provenance_statuses"]
                ),
                "source_limitations": _list_cell(source_metadata["source_limitations"]),
                "source_warnings": _list_cell(source_metadata["source_warnings"]),
                "source_caveats": _list_cell(source_metadata["source_caveats"]),
                "source_provenance_json": _json_cell(source_metadata["source_provenance"]),
                "question_source_types": _list_cell(source_metadata["source_types"]),
                "question_source_content_scopes": _list_cell(
                    source_metadata["source_content_scopes"]
                ),
                "question_source_acquisition_methods": _list_cell(
                    source_metadata["source_acquisition_methods"]
                ),
                "question_source_provenance_statuses": _list_cell(
                    source_metadata["source_provenance_statuses"]
                ),
                "question_source_limitations": _list_cell(source_metadata["source_limitations"]),
                "question_source_warnings": _list_cell(source_metadata["source_warnings"]),
                "question_source_caveats": _list_cell(source_metadata["source_caveats"]),
                "question_source_provenance_json": _json_cell(
                    source_metadata["source_provenance"]
                ),
                "dataset_title": match.dataset.title,
                "dataset_source": match.dataset.source,
                "dataset_source_types": _list_cell(dataset_source_metadata["source_types"]),
                "dataset_source_content_scopes": _list_cell(
                    dataset_source_metadata["source_content_scopes"]
                ),
                "dataset_source_acquisition_methods": _list_cell(
                    dataset_source_metadata["source_acquisition_methods"]
                ),
                "dataset_source_provenance_statuses": _list_cell(
                    dataset_source_metadata["source_provenance_statuses"]
                ),
                "dataset_source_limitations": _list_cell(
                    dataset_source_metadata["source_limitations"]
                ),
                "dataset_source_warnings": _list_cell(dataset_source_metadata["source_warnings"]),
                "dataset_source_caveats": _list_cell(dataset_source_metadata["source_caveats"]),
                "dataset_source_provenance_json": _json_cell(
                    dataset_source_metadata["source_provenance"]
                ),
                "dataset_capability_categories": _list_cell(
                    capability_metadata["dataset_capability_categories"]
                ),
                "dataset_capability_summary": _list_cell(
                    capability_metadata["dataset_capability_summary"]
                ),
                "direct_capability_support": _list_cell(
                    capability_metadata["direct_capability_support"]
                ),
                "proxy_capability_support": _list_cell(
                    capability_metadata["proxy_capability_support"]
                ),
                "missing_capability_support": _list_cell(
                    capability_metadata["missing_capability_support"]
                ),
                "match_answerability_class": capability_metadata["match_answerability_class"],
                "dataset_capability_caveats": _list_cell(
                    capability_metadata["dataset_capability_caveats"]
                ),
                "required_variables": "; ".join(match.question.required_variables),
                "missing_variables": "; ".join(match.missing_variables),
                "rationale": "; ".join(match.rationale),
                "recommended_design": feasibility.get("recommended_design", ""),
                "feasibility_overall": feasibility.get("overall", ""),
                "feasibility_variable_coverage": feasibility.get("variable_coverage", ""),
                "feasibility_population_fit": feasibility.get("population_fit", ""),
                "feasibility_sample_adequacy": feasibility.get("sample_adequacy", ""),
                "feasibility_longitudinal_fit": feasibility.get("longitudinal_fit", ""),
                "feasibility_assay_fit": feasibility.get("assay_fit", ""),
                "governance_reuse_score": governance.get("reuse_score", ""),
                "governance_access_score": governance.get("access_score", ""),
                "governance_license_score": governance.get("license_score", ""),
                "governance_privacy_score": governance.get("privacy_score", ""),
                "governance_risk_flags": "; ".join(risk_flags),
                "score_components_json": _json_cell(score),
                "assessments_json": _json_cell(match.assessments),
                # Blank fields are reserved for later labels, separated from model scores.
                "match_relevance": "",
                "expert_question_quality": "",
                "expert_data_match_quality": "",
                "expert_notes": "",
            }
        )
    return rows


def match_review_records(matches: Iterable[MatchCandidate]) -> list[dict]:
    """Return JSONL review records with full nested match context preserved."""

    records: list[dict] = []
    for rank, match in enumerate(matches, 1):
        source_metadata = _question_source_metadata(match)
        dataset_source_metadata = _dataset_source_metadata(match)
        capability_metadata = _capability_support_metadata(match)
        records.append(
            {
                "rank": rank,
                "match_id": match.match_id,
                "question_id": match.question.question_id,
                "dataset_id": match.dataset.dataset_id,
                "score": match.score.combined,
                "question": match.question.question,
                "primary_source_id": source_metadata["primary_source_id"],
                "source_ids": source_metadata["source_ids"],
                "document_id": source_metadata["document_id"],
                "document_ids": source_metadata["document_ids"],
                "evidence_dois": source_metadata["evidence_dois"],
                "evidence_titles": source_metadata["evidence_titles"],
                "evidence_sections": source_metadata["evidence_sections"],
                "evidence_sentence_indices": source_metadata["evidence_sentence_indices"],
                "source_provenance": source_metadata["source_provenance"],
                "source_types": source_metadata["source_types"],
                "source_content_scopes": source_metadata["source_content_scopes"],
                "source_acquisition_methods": source_metadata["source_acquisition_methods"],
                "source_provenance_statuses": source_metadata["source_provenance_statuses"],
                "source_limitations": source_metadata["source_limitations"],
                "source_warnings": source_metadata["source_warnings"],
                "source_caveats": source_metadata["source_caveats"],
                "question_source_provenance": source_metadata["source_provenance"],
                "question_source_types": source_metadata["source_types"],
                "question_source_content_scopes": source_metadata["source_content_scopes"],
                "question_source_acquisition_methods": source_metadata[
                    "source_acquisition_methods"
                ],
                "question_source_provenance_statuses": source_metadata[
                    "source_provenance_statuses"
                ],
                "question_source_limitations": source_metadata["source_limitations"],
                "question_source_warnings": source_metadata["source_warnings"],
                "question_source_caveats": source_metadata["source_caveats"],
                "dataset_title": match.dataset.title,
                "dataset_source": match.dataset.source,
                "dataset_source_provenance": dataset_source_metadata["source_provenance"],
                "dataset_source_types": dataset_source_metadata["source_types"],
                "dataset_source_content_scopes": dataset_source_metadata[
                    "source_content_scopes"
                ],
                "dataset_source_acquisition_methods": dataset_source_metadata[
                    "source_acquisition_methods"
                ],
                "dataset_source_provenance_statuses": dataset_source_metadata[
                    "source_provenance_statuses"
                ],
                "dataset_source_limitations": dataset_source_metadata["source_limitations"],
                "dataset_source_warnings": dataset_source_metadata["source_warnings"],
                "dataset_source_caveats": dataset_source_metadata["source_caveats"],
                "dataset_capability_categories": capability_metadata[
                    "dataset_capability_categories"
                ],
                "dataset_capability_summary": capability_metadata["dataset_capability_summary"],
                "direct_capability_support": capability_metadata["direct_capability_support"],
                "proxy_capability_support": capability_metadata["proxy_capability_support"],
                "missing_capability_support": capability_metadata["missing_capability_support"],
                "match_answerability_class": capability_metadata["match_answerability_class"],
                "dataset_capability_caveats": capability_metadata["dataset_capability_caveats"],
                "rationale": match.rationale,
                "missing_variables": match.missing_variables,
                "score_components": match.score.to_dict(),
                "assessments": match.assessments,
                "match": match.to_dict(),
                # Top-level blanks keep label ingestion simple for annotation tools.
                "match_relevance": "",
                "expert_question_quality": "",
                "expert_data_match_quality": "",
                "expert_notes": "",
            }
        )
    return records


def export_review_csv(matches: Iterable[MatchCandidate], path: str | Path) -> None:
    """Write a CSV review sheet for human reviewers."""

    rows = match_review_rows(matches)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve a stable header even when no matches pass ranking.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def export_review_jsonl(matches: Iterable[MatchCandidate], path: str | Path) -> None:
    """Write a JSONL review sheet for programmatic annotation tools."""

    write_jsonl(path, match_review_records(matches))


def load_review_labels(path: str | Path) -> list[dict]:
    """Load labels from CSV or JSONL review files."""

    path = Path(path)
    # File suffix is the only format switch to keep the CLI simple.
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _review_value(row: dict, key: str, default: object = "") -> object:
    """Read a top-level review value with nested-match fallback."""

    if key in row:
        return row.get(key, default)
    match = row.get("match", {})
    if key == "match_id":
        return match.get("match_id", default)
    if key == "question_id":
        return match.get("question", {}).get("question_id", default)
    if key == "dataset_id":
        return match.get("dataset", {}).get("dataset_id", default)
    return default


def _first_review_field(row: dict, *keys: str) -> object:
    """Return the first nonblank value from a canonical field or alias."""

    for key in keys:
        value = row.get(key, "")
        if str(value).strip():
            return value
    return ""


def _review_float(row: dict, key: str, *aliases: str) -> float | None:
    """Parse optional numeric reviewer fields."""

    raw = str(_first_review_field(row, key, *aliases)).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _normalized_relevance(row: dict) -> float | None:
    """Normalize binary or 1..5 relevance labels to 0..1."""

    value = _review_float(row, "match_relevance", "expert_match_relevance", "expert_relevance")
    if value is None:
        return None
    if value > 1:
        return min(5.0, value) / 5.0
    return max(0.0, min(1.0, value))


def _row_annotator_id(row: dict, default: str = "") -> str:
    """Prefer row-level reviewer metadata over a CLI/default annotator."""

    for field in ("_effective_annotator_id", "annotator_id", "reviewer_id", "expert_id"):
        value = str(row.get(field, "") or "").strip()
        if value:
            return value
    return str(default or "").strip()


def _metadata_list(value: object) -> list[str]:
    """Normalize list-like or semicolon-delimited provenance values."""

    if isinstance(value, (list, tuple, set)):
        return _clean_review_list(value)
    text = str(value or "").strip()
    if not text:
        return []
    return _clean_review_list(part.strip() for part in text.split(";"))


def _metadata_scalar(row: dict, *fields: str) -> str:
    """Return the first nonblank scalar provenance field."""

    for field in fields:
        value = str(row.get(field, "") or "").strip()
        if value:
            return value
    return ""


def _source_metadata(row: dict) -> dict:
    """Preserve optional document/source IDs for grouped training splits."""

    metadata: dict[str, object] = {}
    for field in ("source_id", "primary_source_id", "document_id"):
        value = _metadata_scalar(row, field)
        if value:
            metadata[field] = value
    for field in (
        "source_ids",
        "document_ids",
        "evidence_dois",
        "evidence_titles",
        "evidence_sections",
        "evidence_sentence_indices",
        "source_types",
        "source_content_scopes",
        "source_acquisition_methods",
        "source_provenance_statuses",
        "source_limitations",
        "source_warnings",
        "source_caveats",
    ):
        values = _metadata_list(row.get(field, ""))
        if values:
            metadata[field] = values
    source_provenance = row.get("source_provenance", [])
    if isinstance(source_provenance, list) and source_provenance:
        metadata["source_provenance"] = source_provenance
    elif isinstance(source_provenance, dict) and source_provenance:
        metadata["source_provenance"] = [source_provenance]
    elif row.get("source_provenance_json"):
        parsed_provenance = _parse_source_provenance_json(row.get("source_provenance_json", ""))
        if parsed_provenance:
            metadata["source_provenance"] = parsed_provenance
    for prefix in ("question", "dataset"):
        prefix_key = f"{prefix}_source"
        for field in (
            "types",
            "content_scopes",
            "acquisition_methods",
            "provenance_statuses",
            "limitations",
            "warnings",
            "caveats",
        ):
            review_field = f"{prefix_key}_{field}"
            values = _metadata_list(row.get(review_field, ""))
            if values:
                metadata[review_field] = values
        structured_field = f"{prefix_key}_provenance"
        provenance = row.get(structured_field, [])
        if isinstance(provenance, list) and provenance:
            metadata[structured_field] = [
                item for item in provenance if isinstance(item, dict) and item
            ]
        elif isinstance(provenance, dict) and provenance:
            metadata[structured_field] = [provenance]
        parsed_provenance = _parse_source_provenance_json(
            row.get(f"{prefix_key}_provenance_json", "")
        )
        if parsed_provenance:
            metadata[structured_field] = parsed_provenance

    match = row.get("match", {})
    question = match.get("question", {}) if isinstance(match, dict) else {}
    if isinstance(question, dict):
        # JSONL review exports may carry richer provenance than the flattened CSV cells.
        question_metadata = question.get("metadata", {})
        if isinstance(question_metadata, dict) and "source_provenance" not in metadata:
            nested_provenance = _question_provenance_records(question_metadata)
            if nested_provenance:
                metadata["source_provenance"] = nested_provenance
        source_ids = _metadata_list(question.get("source_ids", ""))
        evidence = question.get("evidence", [])
        if isinstance(evidence, list):
            source_ids.extend(
                item.get("source_id", "")
                for item in evidence
                if isinstance(item, dict) and item.get("source_id", "")
            )
            evidence_fields = {
                "evidence_dois": "doi",
                "evidence_titles": "title",
                "evidence_sections": "section",
                "evidence_sentence_indices": "sentence_index",
            }
            for metadata_field, evidence_field in evidence_fields.items():
                values = _metadata_list(metadata.get(metadata_field, []))
                values.extend(
                    item.get(evidence_field, "")
                    for item in evidence
                    if isinstance(item, dict) and str(item.get(evidence_field, "")).strip()
                )
                values = _clean_review_list(values)
                if values and metadata_field not in metadata:
                    metadata[metadata_field] = values
        source_ids = _clean_review_list(source_ids)
        if source_ids and "source_ids" not in metadata:
            metadata["source_ids"] = source_ids
        if source_ids and "primary_source_id" not in metadata and "source_id" not in metadata:
            metadata["primary_source_id"] = source_ids[0]
        for field in ("document_id", "document_ids"):
            values = _metadata_list(question.get(field, ""))
            if values and field == "document_ids" and field not in metadata:
                metadata[field] = values
            elif values and field == "document_id" and field not in metadata:
                metadata[field] = values[0]
    dataset = match.get("dataset", {}) if isinstance(match, dict) else {}
    if isinstance(dataset, dict) and "dataset_source_provenance" not in metadata:
        dataset_provenance = provenance_from_record(dataset)
        if dataset_provenance:
            metadata["dataset_source_provenance"] = dataset_provenance
    return metadata


def _parse_source_provenance_json(value: object) -> list[dict]:
    """Parse nested provenance preserved in flattened CSV exports."""

    try:
        parsed = json.loads(str(value or ""))
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict) and item]
    return []


def _has_training_label(row: dict) -> bool:
    """Return true when a review row contains any expert training label."""

    return any(
        str(row.get(field, "")).strip()
        for field in (
            "match_relevance",
            "expert_match_relevance",
            "expert_relevance",
            "expert_question_quality",
            "expert_data_match_quality",
            "expert_notes",
        )
    )


def review_rows_to_match_labels(
    rows: Iterable[dict], annotator_id: str = "", include_unlabeled: bool = False
) -> list[QuestionDataMatchLabel]:
    """Convert completed review rows into typed question-data match labels."""

    labels: list[QuestionDataMatchLabel] = []
    for row in rows:
        if not include_unlabeled and not _has_training_label(row):
            continue
        metadata = {
            **_source_metadata(row),
            "rank": row.get("rank", ""),
            "score": row.get("score", ""),
            "source_review_file": row.get("_source_review_file", ""),
            "raw_match_relevance": _first_review_field(
                row, "match_relevance", "expert_match_relevance", "expert_relevance"
            ),
        }
        labels.append(
            QuestionDataMatchLabel(
                match_id=str(_review_value(row, "match_id")),
                question_id=str(_review_value(row, "question_id")),
                dataset_id=str(_review_value(row, "dataset_id")),
                annotator_id=_row_annotator_id(row, annotator_id),
                relevance_score=_normalized_relevance(row),
                question_quality_score=_review_float(row, "expert_question_quality"),
                data_match_quality_score=_review_float(row, "expert_data_match_quality"),
                notes=str(row.get("expert_notes", "") or "").strip(),
                metadata=metadata,
            )
        )
    return labels


def review_rows_to_question_quality_scores(
    rows: Iterable[dict], annotator_id: str = "", include_unlabeled: bool = False
) -> list[QuestionQualityScore]:
    """Convert review rows into typed question-quality training labels."""

    scores: list[QuestionQualityScore] = []
    for row in rows:
        quality = _review_float(row, "expert_question_quality")
        if not include_unlabeled and quality is None:
            continue
        metadata = {
            **_source_metadata(row),
            "rank": row.get("rank", ""),
            "match_id": _review_value(row, "match_id"),
            "dataset_id": _review_value(row, "dataset_id"),
            "source_review_file": row.get("_source_review_file", ""),
        }
        scores.append(
            QuestionQualityScore(
                question_id=str(_review_value(row, "question_id")),
                annotator_id=_row_annotator_id(row, annotator_id),
                overall_score=quality,
                notes=str(row.get("expert_notes", "") or "").strip(),
                metadata=metadata,
            )
        )
    return scores


def review_rows_to_training_labels(
    rows: Iterable[dict], annotator_id: str = "", include_unlabeled: bool = False
) -> dict[str, list[dict]]:
    """Return review-derived labels as JSON-compatible training records."""

    rows = list(rows)
    return {
        "question_data_match_labels": [
            label.to_dict()
            for label in review_rows_to_match_labels(rows, annotator_id, include_unlabeled)
        ],
        "question_quality_scores": [
            score.to_dict()
            for score in review_rows_to_question_quality_scores(
                rows, annotator_id, include_unlabeled
            )
        ],
    }


def _numeric_labels(rows: Iterable[dict], field: str) -> list[float]:
    """Collect numeric reviewer labels from a single field."""

    values: list[float] = []
    for row in rows:
        raw = str(row.get(field, "")).strip()
        if not raw:
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def summarize_review_labels(rows: Iterable[dict]) -> dict:
    """Summarize expert labels for active-learning and reporting."""

    rows = list(rows)
    labeled = [
        row
        for row in rows
        if str(
            _first_review_field(row, "match_relevance", "expert_match_relevance", "expert_relevance")
        ).strip()
    ]
    question_quality = _numeric_labels(rows, "expert_question_quality")
    data_match_quality = _numeric_labels(rows, "expert_data_match_quality")
    relevant = 0
    for row in labeled:
        raw_relevance = _first_review_field(
            row, "match_relevance", "expert_match_relevance", "expert_relevance"
        )
        try:
            # Numeric relevance supports 0/1 or graded scores without extra schema.
            relevant += 1 if float(raw_relevance) > 0 else 0
        except ValueError:
            relevant += 1 if str(raw_relevance).lower() in {"yes", "true"} else 0
    return {
        "rows": len(rows),
        "labeled": len(labeled),
        "relevant": relevant,
        "label_coverage": round(len(labeled) / len(rows), 3) if rows else 0.0,
        "relevance_rate": round(relevant / len(labeled), 3) if labeled else 0.0,
        "question_quality_labeled": len(question_quality),
        "mean_question_quality": round(sum(question_quality) / len(question_quality), 3)
        if question_quality
        else 0.0,
        "data_match_quality_labeled": len(data_match_quality),
        "mean_data_match_quality": round(sum(data_match_quality) / len(data_match_quality), 3)
        if data_match_quality
        else 0.0,
    }
