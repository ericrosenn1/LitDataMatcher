"""Local literature-file ingestion for canonical pipeline JSONL inputs.

Ingestion creates reproducible literature records and provenance; extraction,
ranking, and source validation happen downstream.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from .literature_xml import XML_SUFFIXES, xml_to_literature_record
from .provenance import local_file_provenance, parser_caveats, summarize_source_provenance
from .schemas import JsonDict, stable_id
from .storage import read_jsonl, write_jsonl
from .text import normalize_text


TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}
JSONL_SUFFIXES = {".jsonl", ".ndjson"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES | JSONL_SUFFIXES | XML_SUFFIXES
INGESTION_SCHEMA_VERSION = "literature_ingestion_v1"

SECTION_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
}


def ingest_literature_sources(
    inputs: Iterable[str | Path],
    out_path: str | Path,
    recursive: bool = False,
    on_error: str = "raise",
) -> JsonDict:
    """Convert local literature files into the JSONL format consumed by `run`."""

    out_path = Path(out_path)
    records, skipped = load_literature_sources(
        inputs,
        recursive=recursive,
        on_error=on_error,
    )
    write_jsonl(out_path, records)
    manifest = build_ingestion_manifest(inputs, out_path, records, skipped, recursive)
    manifest_path = out_path.with_suffix(".manifest.json")
    report_path = out_path.with_suffix(".ingestion_report.md")
    _write_json(manifest_path, manifest)
    write_ingestion_report(report_path, manifest)
    return {
        "records": len(records),
        "skipped": len(skipped),
        "out": str(out_path),
        "manifest": str(manifest_path),
        "report": str(report_path),
    }


def load_literature_sources(
    inputs: Iterable[str | Path],
    recursive: bool = False,
    on_error: str = "raise",
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Load local files into literature records plus skipped-file diagnostics."""

    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'.")

    records: list[JsonDict] = []
    skipped: list[JsonDict] = []
    for path in discover_literature_files(inputs, recursive=recursive):
        file_meta = source_file_metadata(path)
        try:
            # One file can emit multiple records when JSONL already contains rows.
            file_records = read_literature_file(path, file_meta=file_meta)
            for record in file_records:
                record["source_record_count"] = len(file_records)
                record["source_file_status"] = "ok"
            records.extend(file_records)
        except Exception as exc:
            if on_error == "raise":
                raise
            skipped.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "format": path.suffix.lower().lstrip(".") or "unknown",
                    "sha256": file_meta.get("source_sha256", ""),
                    "size_bytes": file_meta.get("source_size_bytes", 0),
                    "modified_time_utc": file_meta.get("source_modified_time_utc", ""),
                    "status": "skipped",
                    "records_emitted": 0,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return records, skipped


def discover_literature_files(inputs: Iterable[str | Path], recursive: bool = False) -> list[Path]:
    """Return supported files from explicit paths and optional directories."""

    files: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            files.extend(
                item
                for item in path.glob(pattern)
                if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
            )
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
            continue
        raise FileNotFoundError(f"No supported literature file found at {path}")
    return sorted(_dedupe_paths(files), key=lambda item: str(item).lower())


def read_literature_file(path: str | Path, file_meta: JsonDict | None = None) -> list[JsonDict]:
    """Read one supported source file into one or more literature records."""

    path = Path(path)
    file_meta = file_meta or source_file_metadata(path)
    suffix = path.suffix.lower()
    if suffix in JSONL_SUFFIXES:
        return [
            _normalize_existing_record(row, path, index, file_meta)
            for index, row in enumerate(read_jsonl(path), 1)
        ]
    if suffix in TEXT_SUFFIXES:
        extraction_method = "markdown" if suffix in {".md", ".markdown"} else "text"
        return [
            text_to_literature_record(
                path,
                path.read_text(encoding="utf-8", errors="replace"),
                extraction_method=extraction_method,
                file_meta=file_meta,
            )
        ]
    if suffix in PDF_SUFFIXES:
        # PDF text is a deterministic fallback, not structured article parsing.
        return [
            text_to_literature_record(
                path,
                _extract_pdf_text(path),
                extraction_method="pdfminer",
                file_meta=file_meta,
            )
        ]
    if suffix in XML_SUFFIXES:
        return [xml_to_literature_record(path, file_meta=file_meta)]
    raise ValueError(f"Unsupported literature file suffix: {path.suffix}")


def text_to_literature_record(
    path: str | Path,
    text: str,
    extraction_method: str = "text",
    file_meta: JsonDict | None = None,
) -> JsonDict:
    """Build a canonical literature JSONL record from extracted text."""

    path = Path(path)
    file_meta = file_meta or source_file_metadata(path)
    title = infer_title(path, text)
    abstract = infer_abstract(text, title)
    clean_text = str(text or "").strip()
    body_text = extract_body_text(clean_text, title=title, abstract=abstract)
    caveats = parser_caveats(
        extraction_method,
        body_text=body_text,
        abstract=abstract,
        sections=_detected_section_headings(clean_text),
        section_records=[],
        fallback=extraction_method in {"pdfminer", "text", "markdown"},
    )
    # Hash-derived IDs stay stable across runs unless source bytes change.
    source_id = stable_id("source", file_meta["source_sha256"])
    document_id = stable_id("doc", file_meta["source_sha256"])
    return {
        "source_id": source_id,
        "document_id": document_id,
        "title": title,
        "abstract": abstract,
        "text": body_text,
        "doi": "",
        "source_path": str(path),
        "source_name": path.name,
        "source_format": path.suffix.lower().lstrip(".") or "text",
        "source_sha256": file_meta["source_sha256"],
        "source_size_bytes": file_meta["source_size_bytes"],
        "source_modified_time_utc": file_meta["source_modified_time_utc"],
        "source_record_count": 1,
        "source_file_status": "ok",
        "ingestion_method": extraction_method,
        "ingestion_schema_version": INGESTION_SCHEMA_VERSION,
        "source_provenance": local_file_provenance(
            file_meta,
            source_type=_source_type_for_path(path),
            content_scope=_content_scope_for_extraction(extraction_method),
            parser_name=extraction_method,
            status="warning" if caveats["warnings"] else "ok",
            warnings=caveats["warnings"],
            limitations=caveats["limitations"],
            next_handoff="litdatamatcher run",
            metadata={"parser_caveats": caveats},
        ).to_dict(),
    }


def infer_title(path: str | Path, text: str) -> str:
    """Infer a document title from headings, first lines, or filename."""

    path = Path(path)
    for line in _nonempty_lines(text)[:30]:
        candidate = line.strip().lstrip("#").strip()
        if not candidate or _is_section_heading(candidate):
            continue
        if len(candidate) <= 200:
            return normalize_text(candidate)
    return normalize_text(path.stem.replace("_", " ").replace("-", " "))


def infer_abstract(text: str, title: str = "") -> str:
    """Infer a compact abstract from an Abstract section or first paragraph."""

    lines = _nonempty_lines(text)
    for index, line in enumerate(lines):
        if normalize_text(line).lower().strip(":") != "abstract":
            continue
        abstract_lines: list[str] = []
        for next_line in lines[index + 1 :]:
            if _is_section_heading(next_line):
                break
            abstract_lines.append(next_line)
        if abstract_lines:
            return normalize_text(" ".join(abstract_lines))

    paragraphs = [normalize_text(part) for part in str(text or "").split("\n\n")]
    title_norm = normalize_text(title).lower()
    for paragraph in paragraphs:
        if len(paragraph) < 40:
            continue
        if title_norm and paragraph.lower() == title_norm:
            continue
        return paragraph
    return ""


def extract_body_text(text: str, title: str = "", abstract: str = "") -> str:
    """Return source text with obvious title and abstract duplication removed."""

    lines = _nonempty_lines(text)
    body_lines: list[str] = []
    title_norm = normalize_text(title).lower()
    index = 0
    while index < len(lines):
        line = lines[index]
        line_norm = normalize_text(line).lower().strip("#: ")
        if title_norm and line_norm == title_norm:
            index += 1
            continue
        if line_norm == "abstract":
            index += 1
            while index < len(lines) and not _is_section_heading(lines[index]):
                index += 1
            continue
        body_lines.append(line)
        index += 1

    body = "\n".join(body_lines).strip()
    if abstract and normalize_text(body).lower() == normalize_text(abstract).lower():
        return ""
    return body


def build_ingestion_manifest(
    inputs: Iterable[str | Path],
    out_path: str | Path,
    records: list[JsonDict],
    skipped: list[JsonDict],
    recursive: bool,
) -> JsonDict:
    """Summarize an ingestion run for reproducibility review."""

    formats: dict[str, int] = {}
    source_entries = _source_entries(records, skipped)
    for record in records:
        source_format = str(record.get("source_format", "unknown") or "unknown")
        formats[source_format] = formats.get(source_format, 0) + 1
    return {
        "schema_version": INGESTION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": [str(Path(path)) for path in inputs],
        "out": str(Path(out_path)),
        "recursive": bool(recursive),
        "total_discovered_files": len(source_entries),
        "records": len(records),
        "records_written": len(records),
        "skipped": skipped,
        "skipped_files": len(skipped),
        "formats": dict(sorted(formats.items())),
        "source_files": source_entries,
        "provenance_summary": summarize_source_provenance(records),
        "warnings": _ingestion_warnings(records, skipped),
        "source_ids": [record.get("source_id", "") for record in records],
        "document_ids": [record.get("document_id", "") for record in records],
    }


def write_ingestion_report(path: str | Path, manifest: JsonDict) -> Path:
    """Write a human-readable ingestion report."""

    path = Path(path)
    lines = [
        "# Literature Ingestion Report",
        "",
        "## Summary",
        "",
        f"- Created UTC: {manifest.get('created_at_utc', '')}",
        f"- Output JSONL: `{manifest.get('out', '')}`",
        f"- Records written: {manifest.get('records_written', 0)}",
        f"- Skipped files: {manifest.get('skipped_files', 0)}",
        f"- Recursive: {manifest.get('recursive', False)}",
        "",
        "## Inputs",
        "",
    ]
    for item in manifest.get("inputs", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Formats", ""])
    formats = manifest.get("formats", {})
    if formats:
        for key in sorted(formats):
            lines.append(f"- {key}: {formats[key]}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Source Files",
            "",
            "| File | Format | Status | Records | Size bytes | SHA-256 | Modified UTC |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for source in manifest.get("source_files", []):
        digest = str(source.get("sha256", ""))
        lines.append(
            "| "
            + " | ".join(
                (
                    _md(str(source.get("path", ""))),
                    _md(str(source.get("format", ""))),
                    _md(str(source.get("status", ""))),
                    str(source.get("records_emitted", 0)),
                    str(source.get("size_bytes", 0)),
                    _md(digest[:12] + ("..." if len(digest) > 12 else "")),
                    _md(str(source.get("modified_time_utc", ""))),
                )
            )
            + " |"
        )
    lines.extend(["", "## Warnings", ""])
    warnings = manifest.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(["", "## Source Provenance", ""])
    provenance = manifest.get("provenance_summary", {})
    lines.extend(
        [
            f"- Records with provenance: {provenance.get('records_with_provenance', 0)}",
            f"- Source types: {_summary_counts(provenance.get('source_types', {}))}",
            f"- Content scopes: {_summary_counts(provenance.get('content_scopes', {}))}",
            f"- Acquisition methods: {_summary_counts(provenance.get('acquisition_methods', {}))}",
        ]
    )
    limitations = provenance.get("limitations", {})
    if limitations:
        lines.append(f"- Limitations: {_summary_counts(limitations)}")
    warnings_summary = provenance.get("warnings", {})
    if warnings_summary:
        lines.append(f"- Provenance warnings: {_summary_counts(warnings_summary)}")
    caveats_summary = provenance.get("review_caveats", {})
    if caveats_summary:
        lines.append(f"- Reviewer caveats: {_summary_counts(caveats_summary)}")
    lines.extend(["", "## Next Action", ""])
    if manifest.get("records_written", 0):
        lines.append(f"Run `litdatamatcher run --input {manifest.get('out', '')} --out run/full`.")
    else:
        lines.append("Add readable literature files or inspect skipped-file diagnostics.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def source_file_metadata(path: str | Path) -> JsonDict:
    """Return stable filesystem provenance for one source file."""

    path = Path(path)
    stat = path.stat()
    return {
        "source_path": str(path),
        "source_name": path.name,
        "source_format": path.suffix.lower().lstrip(".") or "unknown",
        "source_sha256": file_sha256(path),
        "source_size_bytes": stat.st_size,
        "source_modified_time_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
    }


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest for a source file."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_existing_record(
    row: JsonDict, path: Path, index: int, file_meta: JsonDict
) -> JsonDict:
    """Preserve existing JSONL records while filling ingestion metadata gaps."""

    record = dict(row)
    # JSONL passthrough should enrich missing traceability without overwriting upstream fields.
    text = str(record.get("text", "") or record.get("abstract", "") or record.get("title", ""))
    record.setdefault("title", infer_title(path, text))
    record.setdefault("abstract", infer_abstract(text, str(record.get("title", ""))))
    record.setdefault("text", text)
    record.setdefault("doi", "")
    record.setdefault("source_path", str(path))
    record.setdefault("source_name", path.name)
    record.setdefault("source_format", path.suffix.lower().lstrip(".") or "jsonl")
    record.setdefault("source_sha256", file_meta["source_sha256"])
    record.setdefault("source_size_bytes", file_meta["source_size_bytes"])
    record.setdefault("source_modified_time_utc", file_meta["source_modified_time_utc"])
    record.setdefault("source_record_count", 1)
    record.setdefault("source_file_status", "ok")
    record.setdefault("ingestion_method", "jsonl_passthrough")
    record.setdefault("ingestion_schema_version", INGESTION_SCHEMA_VERSION)
    record.setdefault(
        "source_provenance",
        _jsonl_passthrough_provenance(
            file_meta,
            text=text,
            abstract=str(record.get("abstract", "") or ""),
        ),
    )
    record.setdefault(
        "source_id",
        stable_id(
            "source",
            file_meta["source_sha256"],
            index,
            record.get("doi", ""),
            record.get("title", ""),
        ),
    )
    record.setdefault(
        "document_id",
        stable_id(
            "doc",
            file_meta["source_sha256"],
            index,
            record.get("doi", ""),
            record.get("title", ""),
        ),
    )
    return record


def _extract_pdf_text(path: Path) -> str:
    """Extract PDF text with the optional pdfminer dependency."""

    try:
        from pdfminer.high_level import extract_text
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires the optional dependency pdfminer.six "
            "(install with `python -m pip install -e .[nlp]`)."
        ) from exc
    return str(extract_text(str(path)) or "").strip()


def _nonempty_lines(text: str) -> list[str]:
    """Return stripped nonblank lines from extracted document text."""

    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _is_section_heading(line: str) -> bool:
    """Return true for common biomedical section headings."""

    cleaned = normalize_text(line).lower().strip("#: ")
    return cleaned in SECTION_HEADINGS


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    """De-duplicate paths while preserving their original spelling."""

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _source_entries(records: list[JsonDict], skipped: list[JsonDict]) -> list[JsonDict]:
    """Summarize per-source file status from emitted records and skipped files."""

    by_path: dict[str, JsonDict] = {}
    for record in records:
        path = str(record.get("source_path", ""))
        if not path:
            continue
        entry = by_path.setdefault(
            path,
            {
                "path": path,
                "name": record.get("source_name", ""),
                "format": record.get("source_format", "unknown"),
                "sha256": record.get("source_sha256", ""),
                "size_bytes": record.get("source_size_bytes", 0),
                "modified_time_utc": record.get("source_modified_time_utc", ""),
                "records_emitted": 0,
                "status": "ok",
                "source_type": _record_provenance_field(record, "source_type"),
                "content_scope": _record_provenance_field(record, "content_scope"),
                "acquisition_method": _record_provenance_field(record, "acquisition_method"),
            },
        )
        entry["records_emitted"] = int(entry.get("records_emitted", 0)) + 1
    for item in skipped:
        path = str(item.get("path", ""))
        by_path[path] = {
            "path": path,
            "name": item.get("name", Path(path).name if path else ""),
            "format": item.get("format", "unknown"),
            "sha256": item.get("sha256", ""),
            "size_bytes": item.get("size_bytes", 0),
            "modified_time_utc": item.get("modified_time_utc", ""),
            "records_emitted": item.get("records_emitted", 0),
            "status": item.get("status", "skipped"),
            "reason": item.get("reason", ""),
            "message": item.get("message", ""),
        }
    return [by_path[key] for key in sorted(by_path)]


def _ingestion_warnings(records: list[JsonDict], skipped: list[JsonDict]) -> list[str]:
    """Return corpus-level ingestion warnings for manifest/report review."""

    warnings: list[str] = []
    if skipped:
        warnings.append(f"{len(skipped)} source file(s) were skipped")
    for record in records:
        if not str(record.get("abstract", "")).strip():
            warnings.append(f"{record.get('source_path', 'unknown')} has no inferred abstract")
        provenance = record.get("source_provenance", {})
        if isinstance(provenance, dict):
            for warning in provenance.get("warnings", []) or []:
                warnings.append(f"{record.get('source_path', 'unknown')}: {warning}")
    return sorted(set(warnings))


def _md(value: str) -> str:
    """Escape table-breaking Markdown characters."""

    return value.replace("|", "\\|").replace("\n", " ").strip()


def _source_type_for_path(path: str | Path) -> str:
    """Return a coarse source type for provenance summaries."""

    suffix = Path(path).suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in JSONL_SUFFIXES:
        return "jsonl"
    if suffix in XML_SUFFIXES:
        return "xml"
    return suffix.lstrip(".") or "unknown"


def _content_scope_for_extraction(extraction_method: str) -> str:
    """Return how much scientific content an extraction method likely exposes."""

    method = str(extraction_method or "").lower()
    if method == "pdfminer":
        return "full_text_extracted"
    if method in {"text", "markdown"}:
        return "full_text_local"
    if method == "jsonl_passthrough":
        return "record_passthrough"
    return "unknown"


def _provenance_limitations_for_extraction(extraction_method: str) -> list[str]:
    """Return conservative parser caveats for ingestion provenance."""

    method = str(extraction_method or "").lower()
    if method == "pdfminer":
        return ["PDF text order, captions, tables, and references may be noisy."]
    if method == "jsonl_passthrough":
        return ["Record depth depends on upstream JSONL fields."]
    return []


def _jsonl_passthrough_provenance(file_meta: JsonDict, text: str, abstract: str) -> JsonDict:
    """Return standardized provenance for upstream-prepared JSONL rows."""

    caveats = parser_caveats(
        "jsonl_passthrough",
        body_text=text,
        abstract=abstract,
        sections=[],
        section_records=[],
        fallback=True,
    )
    return local_file_provenance(
        file_meta,
        source_type="jsonl",
        content_scope="record_passthrough",
        parser_name="jsonl_passthrough",
        status="warning" if caveats["warnings"] else "ok",
        warnings=caveats["warnings"],
        limitations=caveats["limitations"],
        next_handoff="litdatamatcher run",
        metadata={"parser_caveats": caveats},
    ).to_dict()


def _detected_section_headings(text: str) -> list[str]:
    """Return common section headings visible in extracted local text."""

    headings: list[str] = []
    for line in _nonempty_lines(text):
        cleaned = normalize_text(line).lower().strip("#: ")
        if cleaned in SECTION_HEADINGS:
            headings.append(cleaned)
    return headings


def _record_provenance_field(record: JsonDict, field: str) -> str:
    """Read one provenance field from a literature record."""

    provenance = record.get("source_provenance", {})
    if isinstance(provenance, dict):
        return str(provenance.get(field, "") or "")
    return ""


def _summary_counts(values: JsonDict) -> str:
    """Render a small count dictionary for Markdown reports."""

    if not values:
        return "none"
    return ", ".join(f"{key}: {values[key]}" for key in sorted(values))


def _write_json(path: str | Path, payload: JsonDict) -> None:
    """Write stable pretty JSON for ingestion manifests."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
