"""Helpers for source provenance metadata and review-facing summaries.

This module records how an input reached the pipeline; it does not validate
scientific completeness, license status, or whether a source can answer a
question.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from .schemas import JsonDict, SourceProvenance

PROVENANCE_SCHEMA_VERSION = "source_provenance_v1"

# Caveat codes are review vocabulary for parsers, not strict validation rules.
PARSER_CAVEAT_RECORDS: dict[str, JsonDict] = {
    "abstract_only": {
        "kind": "warning",
        "message": "Only abstract-level text was available; full methods and limitations may be absent.",
        "applies_to": ["pubmed", "openalex", "local_text", "xml"],
    },
    "missing_body_text": {
        "kind": "warning",
        "message": "No body text was extracted; question extraction may only see title or abstract.",
        "applies_to": ["jats", "grobid_tei", "generic_xml", "pdfminer", "text", "markdown", "jsonl_passthrough"],
    },
    "missing_abstract": {
        "kind": "warning",
        "message": "No abstract was available or inferred.",
        "applies_to": ["jats", "grobid_tei", "generic_xml", "pdfminer", "text", "markdown", "jsonl_passthrough"],
    },
    "weak_sections": {
        "kind": "warning",
        "message": "Section structure is weak or absent; section-level evidence context may be shallow.",
        "applies_to": ["jats", "grobid_tei", "generic_xml", "pdfminer", "text", "markdown", "jsonl_passthrough"],
    },
    "generic_xml_fallback": {
        "kind": "warning",
        "message": "Generic XML fallback was used; metadata and section recovery may be incomplete.",
        "applies_to": ["generic_xml"],
    },
    "fallback_parsing": {
        "kind": "warning",
        "message": "Fallback parsing was used; metadata and section recovery may be incomplete.",
        "applies_to": ["text", "markdown", "jsonl_passthrough"],
    },
    "partial_parse_quality": {
        "kind": "warning",
        "message": "Parsed content appears short or shallow and should be reviewed before scale-up.",
        "applies_to": ["jats", "grobid_tei", "generic_xml", "pdfminer", "text", "markdown", "jsonl_passthrough"],
    },
    "pdf_parse_partial": {
        "kind": "limitation",
        "message": "PDF text order, columns, captions, tables, and references may be noisy.",
        "applies_to": ["pdfminer"],
    },
    "no_offsets": {
        "kind": "limitation",
        "message": "Parser does not preserve source character offsets for evidence-span review.",
        "applies_to": ["jats", "grobid_tei", "generic_xml", "pdfminer", "text", "markdown", "jsonl_passthrough"],
    },
    "no_figures_tables": {
        "kind": "limitation",
        "message": "Figures, tables, captions, and supplementary files are not parsed as structured evidence.",
        "applies_to": ["jats", "grobid_tei", "generic_xml", "pdfminer", "text", "markdown", "jsonl_passthrough"],
    },
    "no_reference_filtering": {
        "kind": "limitation",
        "message": "Reference lists may remain in extracted text unless the source parser removed them.",
        "applies_to": ["generic_xml", "pdfminer", "text", "markdown"],
    },
    "jats_not_schema_validated": {
        "kind": "limitation",
        "message": "JATS structure is parsed heuristically and is not yet schema-validated.",
        "applies_to": ["jats"],
    },
    "grobid_upstream_quality": {
        "kind": "limitation",
        "message": "TEI quality depends on upstream GROBID PDF parsing and service configuration.",
        "applies_to": ["grobid_tei"],
    },
    "generic_xml_shallow": {
        "kind": "limitation",
        "message": "Generic XML fallback may miss article-specific metadata and section structure.",
        "applies_to": ["generic_xml"],
    },
    "jsonl_upstream_depth": {
        "kind": "limitation",
        "message": "Record depth depends on upstream JSONL fields and provenance.",
        "applies_to": ["jsonl_passthrough"],
    },
}

PARSER_LIMITATION_MESSAGES: dict[str, str] = {
    code: str(record["message"])
    for code, record in PARSER_CAVEAT_RECORDS.items()
    if record.get("kind") == "limitation"
}

PARSER_WARNING_MESSAGES: dict[str, str] = {
    code: str(record["message"])
    for code, record in PARSER_CAVEAT_RECORDS.items()
    if record.get("kind") == "warning"
}

# Profiles prevent metadata sources from being interpreted as full-text or analysis-ready data.
SOURCE_INTERPRETATION_PROFILES: dict[str, JsonDict] = {
    "text": {
        "source_type": "text",
        "category": "local_literature",
        "content_scope": "local text; full only if the file contains full text",
        "acquisition_method": "local_file",
        "native_id_fields": ["local_path", "source_sha256"],
        "strengths": ["hashable local input", "simple reproducible smoke-test source"],
        "limitations": ["article structure is inferred from plain text"],
        "review_caveats": ["Confirm whether the local text is full article text or a partial excerpt."],
        "do_not_infer": ["publication status", "complete article structure"],
    },
    "markdown": {
        "source_type": "markdown",
        "category": "local_literature",
        "content_scope": "local Markdown; full only if the file contains full text",
        "acquisition_method": "local_file",
        "native_id_fields": ["local_path", "source_sha256"],
        "strengths": ["readable local input", "headings may help section inference"],
        "limitations": ["Markdown headings may reflect notes rather than article sections"],
        "review_caveats": ["Treat Markdown section structure as curator-provided, not validated article structure."],
        "do_not_infer": ["peer-reviewed full text from formatting alone"],
    },
    "pdf": {
        "source_type": "pdf",
        "category": "local_literature",
        "content_scope": "extracted PDF text",
        "acquisition_method": "local_file",
        "native_id_fields": ["local_path", "source_sha256"],
        "strengths": ["works when XML or TEI is unavailable"],
        "limitations": ["text order and tables/captions can be noisy", "no OCR"],
        "review_caveats": ["Review PDF-derived questions for extraction noise and missing table/figure context."],
        "do_not_infer": ["clean structured full text"],
    },
    "jats": {
        "source_type": "jats",
        "category": "structured_literature",
        "content_scope": "structured full text when article body is present",
        "acquisition_method": "local_file",
        "native_id_fields": ["doi", "pmcid", "pmid", "local_path", "source_sha256"],
        "strengths": ["article metadata and sections often preserved"],
        "limitations": ["schema validation and exact span offsets are not yet implemented"],
        "review_caveats": ["Use JATS sections as strong context, but not as exact evidence-span provenance."],
        "do_not_infer": ["license compliance", "schema validity"],
    },
    "grobid_tei": {
        "source_type": "grobid_tei",
        "category": "structured_literature",
        "content_scope": "GROBID TEI text and metadata",
        "acquisition_method": "local_file",
        "native_id_fields": ["doi", "pmcid", "pmid", "local_path", "source_sha256"],
        "strengths": ["can recover structure from PDFs"],
        "limitations": ["quality depends on upstream GROBID output", "exact offsets are not preserved"],
        "review_caveats": ["Inspect TEI-derived questions for upstream PDF parsing artifacts."],
        "do_not_infer": ["that LitDataMatcher managed or validated the GROBID service"],
    },
    "generic_xml": {
        "source_type": "generic_xml",
        "category": "fallback_literature",
        "content_scope": "partial XML text",
        "acquisition_method": "local_file",
        "native_id_fields": ["local_path", "source_sha256"],
        "strengths": ["conservative fallback for unknown XML"],
        "limitations": ["may miss metadata, sections, and article IDs"],
        "review_caveats": ["Treat generic XML output as triage-level until a source-specific parser exists."],
        "do_not_infer": ["structured full text equivalence"],
    },
    "jsonl": {
        "source_type": "jsonl",
        "category": "prepared_literature",
        "content_scope": "record passthrough",
        "acquisition_method": "local_file",
        "native_id_fields": ["source_id", "document_id", "local_path", "source_sha256"],
        "strengths": ["preserves prepared rows and upstream identifiers"],
        "limitations": ["depth depends on upstream fields"],
        "review_caveats": ["Check upstream provenance before treating JSONL rows as full text."],
        "do_not_infer": ["full text unless text and provenance establish it"],
    },
    "pubmed": {
        "source_type": "pubmed",
        "category": "literature_metadata",
        "content_scope": "abstract plus metadata, or metadata only",
        "acquisition_method": "ncbi_eutilities",
        "native_id_fields": ["pmid", "doi", "pmcid"],
        "strengths": ["PMID/DOI tracking", "citation metadata", "abstract-level triage"],
        "limitations": ["not full text", "abstract availability varies"],
        "review_caveats": ["PubMed records should be treated as abstract-level evidence unless full text is separately linked."],
        "do_not_infer": ["full methods", "full limitations", "complete future-work context"],
    },
    "openalex": {
        "source_type": "openalex",
        "category": "scholarly_metadata",
        "content_scope": "metadata plus reconstructed abstract when available",
        "acquisition_method": "openalex_api",
        "native_id_fields": ["openalex_work_id", "doi"],
        "strengths": ["broad scholarly discovery", "citation/concept metadata"],
        "limitations": ["reconstructed abstracts are not article body text"],
        "review_caveats": ["OpenAlex records are discovery metadata and should not be treated as article-body extraction."],
        "do_not_infer": ["full article evidence"],
    },
    "europepmc": {
        "source_type": "europepmc",
        "category": "literature_metadata",
        "content_scope": "abstract plus metadata, or metadata only",
        "acquisition_method": "europepmc_rest_api",
        "native_id_fields": ["europepmc_source", "europepmc_id", "pmid", "pmcid", "doi"],
        "strengths": ["Europe PMC identifiers", "publication metadata", "open-access status metadata"],
        "limitations": ["search response is not article body text", "version relations require source-specific review"],
        "review_caveats": ["Europe PMC search rows are discovery metadata unless full text is separately acquired and parsed."],
        "do_not_infer": ["full methods", "full article evidence", "license clearance from metadata alone"],
    },
    "crossref": {
        "source_type": "crossref",
        "category": "scholarly_metadata",
        "content_scope": "DOI metadata only",
        "acquisition_method": "crossref_works_api",
        "native_id_fields": ["doi"],
        "strengths": ["DOI normalization", "publication update and relation metadata"],
        "limitations": ["not article body text", "metadata quality varies by registrant"],
        "review_caveats": ["Crossref rows support DOI-level linking and version review, not scientific evidence extraction by themselves."],
        "do_not_infer": ["full article evidence", "current retraction or correction status without source review"],
    },
    "clinicaltrials": {
        "source_type": "clinicaltrials",
        "category": "registry_metadata",
        "content_scope": "trial registry metadata",
        "acquisition_method": "clinicaltrials_api",
        "native_id_fields": ["nct_id"],
        "strengths": ["intervention, condition, design, and outcome metadata"],
        "limitations": ["not patient-level data", "reuse rights require study-specific review"],
        "review_caveats": ["ClinicalTrials.gov metadata can suggest a study, not guarantee usable individual-level data."],
        "do_not_infer": ["patient-level access", "analysis-ready trial data"],
    },
    "geo": {
        "source_type": "geo",
        "category": "dataset_metadata",
        "content_scope": "GEO study summary metadata",
        "acquisition_method": "ncbi_eutilities",
        "native_id_fields": ["geo_accession", "gds_uid"],
        "strengths": ["omics accession discovery", "study-level summary"],
        "limitations": ["variable-level metadata requires file inspection"],
        "review_caveats": ["GEO summaries identify candidate datasets; inspect files before judging variable availability."],
        "do_not_infer": ["downloaded matrices", "analysis-ready expression data"],
    },
    "mgnify": {
        "source_type": "mgnify",
        "category": "dataset_metadata",
        "content_scope": "microbiome study metadata",
        "acquisition_method": "mgnify_api_v2",
        "native_id_fields": ["mgnify_accession", "study_accession"],
        "strengths": ["microbiome study discovery", "sample-count metadata when available"],
        "limitations": ["sample tables and detailed biome metadata may need follow-up"],
        "review_caveats": ["MGnify list records are study metadata, not automatically usable sample tables."],
        "do_not_infer": ["analysis-ready microbiome feature tables"],
    },
    "curated_biomedical_catalog": {
        "source_type": "curated_biomedical_catalog",
        "category": "curated_dataset_metadata",
        "content_scope": "offline curated dataset metadata",
        "acquisition_method": "bundled_curated_catalog",
        "native_id_fields": ["dataset_id", "source", "url"],
        "strengths": ["deterministic offline matching", "transparent built-in examples"],
        "limitations": ["not a live repository snapshot", "source-specific validation is still required"],
        "review_caveats": ["Default offline catalog metadata should be verified against source repositories before publication use."],
        "do_not_infer": ["live database coverage", "downloaded datasets", "analysis-ready variables"],
    },
    "capability_registry": {
        "source_type": "capability_registry",
        "category": "derived_catalog",
        "content_scope": "observed and plausibly derived capability catalog",
        "acquisition_method": "local_derivation_catalog",
        "native_id_fields": ["dataset_id", "variable_name", "derivation_rule_id"],
        "strengths": ["makes possible derivations inspectable before analysis"],
        "limitations": ["derived capabilities are not computed analyses"],
        "review_caveats": ["Capability records describe plausible availability, not completed derivation or validation."],
        "do_not_infer": ["computed variables", "statistical evidence"],
    },
}

ADAPTER_SOURCE_TYPES = ("pubmed", "openalex", "europepmc", "crossref", "clinicaltrials", "geo", "mgnify")

# Module boundaries are developer-facing documentation, not runtime dispatch rules.
MODULE_BOUNDARIES: dict[str, JsonDict] = {
    "litdatamatcher.ingestion": {
        "responsibility": "Convert local files into pipeline-ready literature records with provenance.",
        "inputs": ["text", "Markdown", "PDF", "JSONL/NDJSON", "XML"],
        "outputs": ["literature JSONL records", "ingestion manifest", "ingestion report"],
        "does_not_do": ["question extraction", "ranking", "license validation"],
    },
    "litdatamatcher.literature_xml": {
        "responsibility": "Parse JATS/PMC, GROBID TEI, or generic XML into literature records.",
        "inputs": ["XML file bytes", "source file metadata"],
        "outputs": ["title/abstract/body text", "article metadata", "section records"],
        "does_not_do": ["schema validation", "character offsets", "figure/table extraction"],
    },
    "litdatamatcher.grobid": {
        "responsibility": "Call an optional external GROBID service to produce TEI XML.",
        "inputs": ["local PDF", "GROBID service URL"],
        "outputs": ["TEI XML file", "conversion metadata"],
        "does_not_do": ["literature record creation", "question extraction", "service management"],
    },
    "litdatamatcher.adapters": {
        "responsibility": "Retrieve optional source metadata and normalize it into literature or dataset records.",
        "inputs": ["query", "cached/live HTTP response"],
        "outputs": ["literature-like rows", "DatasetRecord objects"],
        "does_not_do": ["full dataset download", "computed analysis", "full-text parsing"],
    },
    "litdatamatcher.provenance": {
        "responsibility": "Build, summarize, interpret, and check source-provenance transfer.",
        "inputs": ["provenance dictionaries", "records at pipeline handoffs"],
        "outputs": ["source summaries", "review caveats", "transfer-check reports"],
        "does_not_do": ["scientific validation", "license adjudication"],
    },
    "litdatamatcher.literature": {
        "responsibility": "Extract deterministic question candidates and copy source provenance into question metadata.",
        "inputs": ["literature records"],
        "outputs": ["QuestionCandidate objects"],
        "does_not_do": ["gold-standard adjudication", "model training"],
    },
    "litdatamatcher.ranking": {
        "responsibility": "Score question-to-dataset matches with inspectable components.",
        "inputs": ["QuestionCandidate objects", "DatasetRecord objects", "EvidenceSynthesis index"],
        "outputs": ["MatchCandidate objects"],
        "does_not_do": ["causal inference", "statistical analysis of raw datasets"],
    },
    "litdatamatcher.review": {
        "responsibility": "Export ranked matches for human review and normalize completed labels.",
        "inputs": ["MatchCandidate objects", "completed review CSV/JSONL"],
        "outputs": ["review CSV/JSONL", "training-label objects"],
        "does_not_do": ["label adjudication by itself", "proof of match correctness"],
    },
    "litdatamatcher.reporting": {
        "responsibility": "Summarize run artifacts and caveats for inspection.",
        "inputs": ["run JSONL artifacts", "SQLite tables when present"],
        "outputs": ["publication-style Markdown report"],
        "does_not_do": ["publication validation", "external source validation"],
    },
}

MODULE_OWNERSHIP_REGISTRY: dict[str, JsonDict] = {
    "ingestion": {
        "owner_module": "litdatamatcher.ingestion",
        "owned_by": "local source ingestion node",
        "responsibility": "local file discovery, record creation, ingestion manifest/report",
    },
    "xml_parsing": {
        "owner_module": "litdatamatcher.literature_xml",
        "owned_by": "structured literature parser node",
        "responsibility": "JATS/PMC, GROBID TEI, and generic XML parsing",
    },
    "grobid_bridge": {
        "owner_module": "litdatamatcher.grobid",
        "owned_by": "optional external-service bridge",
        "responsibility": "PDF-to-TEI conversion through a running GROBID service",
    },
    "live_adapters": {
        "owner_module": "litdatamatcher.adapters",
        "owned_by": "optional metadata adapter layer",
        "responsibility": "PubMed/OpenAlex/Europe PMC/Crossref/ClinicalTrials.gov/GEO/MGnify metadata retrieval and normalization",
    },
    "provenance": {
        "owner_module": "litdatamatcher.provenance",
        "owned_by": "source traceability layer",
        "responsibility": "provenance construction, source profiles, caveats, summaries, and transfer checks",
    },
    "question_extraction": {
        "owner_module": "litdatamatcher.literature",
        "owned_by": "literature question node",
        "responsibility": "deterministic candidate question extraction and provenance copy into question metadata",
    },
    "matching": {
        "owner_module": "litdatamatcher.ranking",
        "owned_by": "question-to-data matching node",
        "responsibility": "ranked MatchCandidate construction and component scores",
    },
    "review_export": {
        "owner_module": "litdatamatcher.review",
        "owned_by": "review and label handoff layer",
        "responsibility": "CSV/JSONL review exports and completed-review label normalization",
    },
    "reporting": {
        "owner_module": "litdatamatcher.reporting",
        "owned_by": "inspection/report layer",
        "responsibility": "publication-style summaries and provenance caveat presentation",
    },
}


def utc_now() -> str:
    """Return a compact UTC timestamp for source retrieval metadata."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def module_boundary_map() -> JsonDict:
    """Return the inspectable module-boundary inventory for run artifacts."""

    return {key: dict(value) for key, value in MODULE_BOUNDARIES.items()}


def module_ownership_registry() -> JsonDict:
    """Return responsibilities keyed by the module that owns each boundary."""

    return {key: dict(value) for key, value in MODULE_OWNERSHIP_REGISTRY.items()}


def adapter_source_profile_table() -> JsonDict:
    """Return compact source profiles for optional external adapters."""

    return {source_type: source_profile(source_type) for source_type in ADAPTER_SOURCE_TYPES}


def parser_caveat_records() -> list[JsonDict]:
    """Return standardized parser caveat records for developer inspection."""

    return [
        {"code": code, **dict(record)}
        for code, record in sorted(PARSER_CAVEAT_RECORDS.items())
    ]


def source_profile(source_type: str) -> JsonDict:
    """Return a reviewer-facing source profile for a known source type."""

    normalized = str(source_type or "unknown").strip().lower()
    profile = SOURCE_INTERPRETATION_PROFILES.get(normalized)
    if profile:
        return dict(profile)
    return {
        "source_type": normalized or "unknown",
        "category": "unknown",
        "content_scope": "unknown",
        "strengths": [],
        "limitations": ["Source type is not yet profiled."],
        "review_caveats": ["Unprofiled source type; inspect source provenance before interpretation."],
        "do_not_infer": ["validated content depth"],
    }


def parser_caveats(
    parser_name: str,
    *,
    body_text: str = "",
    abstract: str = "",
    sections: Iterable[object] | None = None,
    section_records: Iterable[JsonDict] | None = None,
    fallback: bool = False,
) -> JsonDict:
    """Return standardized parser warnings, limitations, and quality metadata."""

    parser = str(parser_name or "unknown").strip().lower()
    sections = list(sections or [])
    section_records = list(section_records or [])
    body = str(body_text or "").strip()
    abstract_text = str(abstract or "").strip()
    warning_codes: list[str] = []
    limitation_codes = ["no_offsets", "no_figures_tables"]

    if not body:
        warning_codes.append("missing_body_text")
        if abstract_text:
            warning_codes.append("abstract_only")
    elif len(body) < 300:
        warning_codes.append("partial_parse_quality")
    if not abstract_text:
        warning_codes.append("missing_abstract")
    if len(sections) < 1 or not section_records:
        warning_codes.append("weak_sections")
    if parser == "generic_xml":
        warning_codes.append("generic_xml_fallback")
    elif fallback or parser == "jsonl_passthrough":
        warning_codes.append("fallback_parsing")

    if parser == "pdfminer":
        limitation_codes.extend(["pdf_parse_partial", "no_reference_filtering"])
    elif parser == "jats":
        limitation_codes.extend(["jats_not_schema_validated"])
    elif parser == "grobid_tei":
        limitation_codes.extend(["grobid_upstream_quality"])
    elif parser == "generic_xml":
        limitation_codes.extend(["generic_xml_shallow", "no_reference_filtering"])
    elif parser == "jsonl_passthrough":
        limitation_codes.extend(["jsonl_upstream_depth"])
    else:
        limitation_codes.append("no_reference_filtering")

    warning_codes = _dedupe_strings(warning_codes)
    limitation_codes = _dedupe_strings(limitation_codes)
    return {
        "parser_name": parser,
        "parse_quality": _parse_quality(warning_codes, body),
        "warning_codes": warning_codes,
        "limitation_codes": limitation_codes,
        "warning_records": _selected_caveat_records(warning_codes),
        "limitation_records": _selected_caveat_records(limitation_codes),
        "warnings": [PARSER_WARNING_MESSAGES[code] for code in warning_codes],
        "limitations": [PARSER_LIMITATION_MESSAGES[code] for code in limitation_codes],
    }


def provenance_review_caveats(provenances: Iterable[JsonDict]) -> list[str]:
    """Translate provenance records into concise reviewer-facing caveats."""

    caveats: list[str] = []
    provenances = list(provenances)
    if not provenances:
        return ["No source provenance was available for this record."]
    for provenance in provenances:
        source_type = str(provenance.get("source_type", "") or "unknown")
        profile = source_profile(source_type)
        caveats.extend(profile.get("review_caveats", []))
        content_scope = str(provenance.get("content_scope", "") or "").lower()
        if "metadata_only" in content_scope or content_scope in {"study_metadata", "dataset_metadata"}:
            caveats.append("Metadata-only records should not be treated as direct evidence of answerability.")
        if "abstract" in content_scope:
            caveats.append("Abstract-level records may miss methods, limitations, and full future-work context.")
        if "derived" in content_scope or source_type == "capability_registry":
            caveats.append("Derived capability records are plausibility catalog entries, not computed analyses.")
        for warning in provenance.get("warnings", []) or []:
            caveats.append(str(warning))
        for limitation in provenance.get("limitations", []) or []:
            caveats.append(str(limitation))
    return _dedupe_strings(caveats)


def provenance_interpretation(provenances: Iterable[JsonDict]) -> JsonDict:
    """Return source-profile and caveat context for review/report surfaces."""

    provenances = list(provenances)
    source_types = _dedupe_strings(item.get("source_type", "unknown") for item in provenances)
    content_scopes = _dedupe_strings(item.get("content_scope", "unknown") for item in provenances)
    return {
        "source_types": source_types,
        "content_scopes": content_scopes,
        "profiles": [source_profile(source_type) for source_type in source_types],
        "caveats": provenance_review_caveats(provenances),
    }


def check_provenance_transfer(
    *,
    source_records: Iterable[JsonDict] | None = None,
    questions: Iterable[JsonDict] | None = None,
    datasets: Iterable[JsonDict] | None = None,
    review_records: Iterable[JsonDict] | None = None,
    report_summary: JsonDict | None = None,
) -> JsonDict:
    """Check whether source provenance survives canonical handoffs."""

    # This diagnostic reports traceability gaps for reviewers; it does not block a run.
    source_records = list(source_records or [])
    questions = list(questions or [])
    datasets = list(datasets or [])
    review_records = list(review_records or [])
    report_summary = dict(report_summary or {})
    stages = {
        "source_records": _stage_counts(source_records),
        "question_metadata": _stage_counts(questions),
        "dataset_records": _stage_counts(datasets),
        "review_records": _stage_counts(review_records),
        "review_visibility": _review_visibility_counts(review_records),
        "report_summary": {
            "records": 1 if report_summary else 0,
            "with_provenance": 1 if report_summary.get("records_with_provenance", 0) else 0,
            "without_provenance": 0 if report_summary.get("records_with_provenance", 0) else (1 if report_summary else 0),
            "provenance_entries": int(report_summary.get("records_with_provenance", 0) or 0),
            "has_review_caveats": bool(report_summary.get("review_caveats")),
            "has_interpretation": bool(report_summary.get("interpretation")),
        },
    }
    issues: list[JsonDict] = []
    issues.extend(_missing_stage_issues("source_record", source_records))
    issues.extend(_missing_stage_issues("question_metadata", questions))
    issues.extend(_missing_stage_issues("dataset_record", datasets))
    issues.extend(_missing_review_issues(review_records))
    if report_summary and "review_caveats" not in report_summary:
        issues.append(
            {
                "severity": "warning",
                "stage": "report_summary",
                "message": "source_provenance_summary.json has no reviewer-facing caveat summary.",
            }
        )
    return {
        "schema_version": "provenance_transfer_check_v1",
        "status": "pass" if not issues else "needs_review",
        "stages": stages,
        "issues": issues,
        "module_ownership": module_ownership_registry(),
        "module_boundaries": module_boundary_map(),
    }


def local_file_provenance(
    file_meta: JsonDict,
    *,
    source_type: str,
    content_scope: str,
    acquisition_method: str = "local_file",
    parser_name: str = "",
    parser_version: str = "",
    status: str = "ok",
    warnings: Iterable[str] | None = None,
    limitations: Iterable[str] | None = None,
    next_handoff: str = "",
    metadata: JsonDict | None = None,
) -> SourceProvenance:
    """Build provenance for a local file that was read into the pipeline."""

    source_path = str(file_meta.get("source_path", "") or "")
    # A provenance entry describes one ingestion event, not necessarily one paper.
    return SourceProvenance(
        source_type=source_type,
        source_locator=source_path,
        source_name=str(file_meta.get("source_name", "") or Path(source_path).name),
        content_scope=content_scope,
        acquisition_method=acquisition_method,
        parser_name=parser_name,
        parser_version=parser_version,
        retrieval_time_utc=utc_now(),
        local_path=source_path,
        source_sha256=str(file_meta.get("source_sha256", "") or ""),
        source_size_bytes=int(file_meta.get("source_size_bytes", 0) or 0),
        source_modified_time_utc=str(file_meta.get("source_modified_time_utc", "") or ""),
        status=status,
        warnings=list(warnings or []),
        limitations=list(limitations or []),
        next_handoff=next_handoff,
        schema_version=PROVENANCE_SCHEMA_VERSION,
        metadata=dict(metadata or {}),
    )


def remote_source_provenance(
    *,
    source_type: str,
    source_url: str,
    adapter_name: str,
    content_scope: str,
    raw_record_id: str = "",
    acquisition_method: str = "api",
    adapter_version: str = "",
    retrieval_time_utc: str = "",
    status: str = "ok",
    warnings: Iterable[str] | None = None,
    limitations: Iterable[str] | None = None,
    next_handoff: str = "",
    metadata: JsonDict | None = None,
) -> SourceProvenance:
    """Build provenance for metadata retrieved from a remote source adapter."""

    # Adapter vocabulary is documented first; code intentionally tolerates new source values.
    return SourceProvenance(
        source_type=source_type,
        source_locator=source_url or raw_record_id,
        source_name=source_type,
        content_scope=content_scope,
        acquisition_method=acquisition_method,
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        retrieval_time_utc=retrieval_time_utc or utc_now(),
        source_url=source_url,
        raw_record_id=raw_record_id,
        status=status,
        warnings=list(warnings or []),
        limitations=list(limitations or []),
        next_handoff=next_handoff,
        schema_version=PROVENANCE_SCHEMA_VERSION,
        metadata=dict(metadata or {}),
    )


def curated_catalog_provenance(
    *,
    dataset_id: str,
    source_name: str,
    source_url: str = "",
) -> SourceProvenance:
    """Build deterministic provenance for bundled curated dataset metadata."""

    return SourceProvenance(
        source_type="curated_biomedical_catalog",
        source_locator=source_url or dataset_id,
        source_name=source_name or "curated biomedical catalog",
        content_scope="dataset_metadata",
        acquisition_method="bundled_curated_catalog",
        adapter_name="CuratedBiomedicalCatalogAdapter",
        adapter_version="static",
        source_url=source_url,
        raw_record_id=dataset_id,
        status="warning",
        warnings=[
            "Offline curated catalog metadata should be checked against the source repository before publication use."
        ],
        limitations=[
            "Catalog variables and counts are curated summaries, not downloaded or analyzed source datasets."
        ],
        next_handoff="dataset matching",
        schema_version=PROVENANCE_SCHEMA_VERSION,
        metadata={"source_profile": source_profile("curated_biomedical_catalog")},
    )


def provenance_dict(provenance: SourceProvenance | JsonDict | None) -> JsonDict:
    """Return provenance as a plain dictionary, tolerating absent metadata."""

    if provenance is None:
        return {}
    if isinstance(provenance, SourceProvenance):
        return provenance.to_dict()
    return dict(provenance)


def attach_source_provenance(record: JsonDict, provenance: SourceProvenance | JsonDict) -> JsonDict:
    """Attach source provenance to a literature or dataset record."""

    record["source_provenance"] = provenance_dict(provenance)
    return record


def provenance_from_record(record: JsonDict) -> list[JsonDict]:
    """Extract one or more provenance dictionaries from a record."""

    values: list[JsonDict] = []
    # Records may carry provenance at top level or inside metadata after node handoffs.
    direct = record.get("source_provenance", {})
    if isinstance(direct, dict) and direct:
        values.append(direct)
    if isinstance(direct, list):
        values.extend(item for item in direct if isinstance(item, dict) and item)
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        nested = metadata.get("source_provenance", {})
        if isinstance(nested, dict) and nested:
            values.append(nested)
        nested_many = metadata.get("source_provenances", [])
        if isinstance(nested_many, list):
            values.extend(item for item in nested_many if isinstance(item, dict) and item)
    return _dedupe_provenance(values)


def summarize_source_provenance(records: Iterable[JsonDict]) -> JsonDict:
    """Summarize source types, content scopes, and caveats across records.

    ``records_with_provenance`` counts provenance entries after de-duplication,
    not guaranteed unique papers, datasets, or scientific entities.
    """

    source_types: Counter[str] = Counter()
    content_scopes: Counter[str] = Counter()
    acquisition_methods: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    limitations: Counter[str] = Counter()
    review_caveats: Counter[str] = Counter()
    total = 0
    inspected = 0
    records_without_provenance = 0
    for record in records:
        inspected += 1
        provenances = provenance_from_record(record)
        if not provenances:
            records_without_provenance += 1
        for provenance in provenances:
            total += 1
            source_types[str(provenance.get("source_type", "unknown") or "unknown")] += 1
            content_scopes[str(provenance.get("content_scope", "unknown") or "unknown")] += 1
            acquisition_methods[
                str(provenance.get("acquisition_method", "unknown") or "unknown")
            ] += 1
            statuses[str(provenance.get("status", "unknown") or "unknown")] += 1
            for warning in provenance.get("warnings", []) or []:
                warnings[str(warning)] += 1
            for limitation in provenance.get("limitations", []) or []:
                limitations[str(limitation)] += 1
            for caveat in provenance_review_caveats([provenance]):
                review_caveats[caveat] += 1
    return {
        "input_records_inspected": inspected,
        "records_with_provenance": total,
        "records_without_provenance": records_without_provenance,
        "source_types": dict(sorted(source_types.items())),
        "content_scopes": dict(sorted(content_scopes.items())),
        "acquisition_methods": dict(sorted(acquisition_methods.items())),
        "statuses": dict(sorted(statuses.items())),
        "warnings": dict(warnings.most_common(10)),
        "limitations": dict(limitations.most_common(10)),
        "review_caveats": dict(review_caveats.most_common(12)),
        "interpretation": [
            "records_with_provenance counts provenance entries, not necessarily unique papers or datasets.",
            "Metadata-only and abstract-level records require source-specific review before publication use.",
            "Warnings and limitations should follow records into review and annotation workflows.",
        ],
    }


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
    """Return nonblank strings with order-preserving deduplication."""

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


def _parse_quality(warning_codes: list[str], body_text: str) -> str:
    """Classify parser quality for provenance metadata."""

    if "missing_body_text" in warning_codes:
        return "metadata_only_or_empty"
    if "generic_xml_fallback" in warning_codes or "fallback_parsing" in warning_codes:
        return "fallback_partial"
    if "weak_sections" in warning_codes or "partial_parse_quality" in warning_codes:
        return "partial_review_needed"
    if len(body_text.strip()) >= 300:
        return "usable_text"
    return "unknown"


def _selected_caveat_records(codes: Iterable[str]) -> list[JsonDict]:
    """Return structured caveat records for selected parser caveat codes."""

    records: list[JsonDict] = []
    for code in codes:
        record = PARSER_CAVEAT_RECORDS.get(code)
        if record:
            records.append({"code": code, **dict(record)})
    return records


def _stage_counts(records: list[JsonDict]) -> JsonDict:
    """Count provenance coverage for one handoff stage."""

    with_provenance = 0
    entries = 0
    for record in records:
        provenances = provenance_from_record(record)
        if provenances:
            with_provenance += 1
            entries += len(provenances)
    return {
        "records": len(records),
        "with_provenance": with_provenance,
        "without_provenance": len(records) - with_provenance,
        "provenance_entries": entries,
    }


def _review_visibility_counts(records: list[JsonDict]) -> JsonDict:
    """Count whether review records expose provenance in reviewable forms."""

    structured = 0
    question_structured = 0
    dataset_structured = 0
    flattened = 0
    question_flattened = 0
    dataset_flattened = 0
    caveats = 0
    json_cells = 0
    question_json_cells = 0
    dataset_json_cells = 0
    for record in records:
        question_visible = _review_side_visible(record, "question")
        dataset_visible = _review_side_visible(record, "dataset")
        if question_visible["structured"] or dataset_visible["structured"]:
            structured += 1
        if question_visible["structured"]:
            question_structured += 1
        if dataset_visible["structured"]:
            dataset_structured += 1
        if question_visible["flat"] or dataset_visible["flat"]:
            flattened += 1
        if question_visible["flat"]:
            question_flattened += 1
        if dataset_visible["flat"]:
            dataset_flattened += 1
        if question_visible["json"] or dataset_visible["json"]:
            json_cells += 1
        if question_visible["json"]:
            question_json_cells += 1
        if dataset_visible["json"]:
            dataset_json_cells += 1
        if record.get("source_caveats") or record.get("question_source_caveats") or record.get("dataset_source_caveats"):
            caveats += 1
    return {
        "records": len(records),
        "structured_provenance_records": structured,
        "question_structured_provenance_records": question_structured,
        "dataset_structured_provenance_records": dataset_structured,
        "flattened_provenance_records": flattened,
        "question_flattened_provenance_records": question_flattened,
        "dataset_flattened_provenance_records": dataset_flattened,
        "provenance_json_cell_records": json_cells,
        "question_provenance_json_cell_records": question_json_cells,
        "dataset_provenance_json_cell_records": dataset_json_cells,
        "source_caveat_records": caveats,
    }


def _missing_stage_issues(stage: str, records: list[JsonDict]) -> list[JsonDict]:
    """Return issue rows for records without provenance at a handoff stage."""

    issues: list[JsonDict] = []
    for index, record in enumerate(records, 1):
        if provenance_from_record(record):
            continue
        issues.append(
            {
                "severity": "warning",
                "stage": stage,
                "record_index": index,
                "record_id": _record_identifier(record),
                "message": "Record has no source provenance at this handoff.",
            }
        )
    return issues


def _missing_review_issues(records: list[JsonDict]) -> list[JsonDict]:
    """Return issue rows for review records missing visible provenance context."""

    issues: list[JsonDict] = []
    for index, record in enumerate(records, 1):
        question_visible = _review_side_visible(record, "question")
        dataset_visible = _review_side_visible(record, "dataset")
        if any(question_visible.values()) or any(dataset_visible.values()):
            continue
        issues.append(
            {
                "severity": "warning",
                "stage": "review_record",
                "record_index": index,
                "record_id": _record_identifier(record),
                "message": "Review record has no structured or flattened provenance context.",
            }
        )
    return issues


def _review_side_visible(record: JsonDict, side: str) -> JsonDict:
    """Return populated review visibility flags for question or dataset provenance."""

    if side == "question":
        structured_fields = ("question_source_provenance", "source_provenance")
        json_fields = ("question_source_provenance_json", "source_provenance_json")
        flat_fields = (
            "question_source_types",
            "question_source_content_scopes",
            "question_source_acquisition_methods",
            "source_types",
            "source_content_scopes",
            "source_acquisition_methods",
        )
        nested_field = "question"
    else:
        structured_fields = ("dataset_source_provenance",)
        json_fields = ("dataset_source_provenance_json",)
        flat_fields = (
            "dataset_source_types",
            "dataset_source_content_scopes",
            "dataset_source_acquisition_methods",
        )
        nested_field = "dataset"
    structured = any(_provenance_value_populated(record.get(field)) for field in structured_fields)
    json_visible = any(_provenance_json_cell_populated(record.get(field)) for field in json_fields)
    flat = any(str(record.get(field, "") or "").strip() for field in flat_fields)
    match = record.get("match", {})
    nested = match.get(nested_field, {}) if isinstance(match, dict) else {}
    nested_structured = bool(provenance_from_record(nested)) if isinstance(nested, dict) else False
    return {
        "structured": structured or nested_structured,
        "json": json_visible,
        "flat": flat,
    }


def _provenance_value_populated(value: object) -> bool:
    """Return true only for non-empty provenance dictionaries."""

    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return any(isinstance(item, dict) and item for item in value)
    return False


def _provenance_json_cell_populated(value: object) -> bool:
    """Return true only for JSON cells containing non-empty provenance records."""

    text = str(value or "").strip()
    if not text:
        return False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    return _provenance_value_populated(parsed)


def _record_identifier(record: JsonDict) -> str:
    """Return the best available stable identifier for diagnostics."""

    for field in (
        "match_id",
        "question_id",
        "dataset_id",
        "document_id",
        "source_id",
        "raw_record_id",
        "title",
    ):
        value = str(record.get(field, "") or "").strip()
        if value:
            return value
    nested_question = record.get("question", {})
    if isinstance(nested_question, dict):
        return str(nested_question.get("question_id", "") or nested_question.get("question", "") or "")
    return ""
