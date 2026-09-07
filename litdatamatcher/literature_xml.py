"""Structured XML extraction for JATS/PMC, GROBID TEI, and generic XML.

The parser preserves article-level sections and metadata where possible, but it
does not yet provide character offsets, table/figure provenance, or validated
evidence-span annotations.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .provenance import local_file_provenance, parser_caveats
from .schemas import JsonDict, stable_id
from .text import normalize_text


XML_SUFFIXES = {".xml", ".nxml", ".jats", ".tei"}


def xml_to_literature_record(path: str | Path, file_meta: JsonDict) -> JsonDict:
    """Parse one structured XML file into a canonical literature record."""

    path = Path(path)
    root = ET.fromstring(path.read_bytes())
    parser = detect_xml_parser(root)
    # Parser choice controls provenance content_scope and downstream caveats.
    if parser == "grobid_tei":
        parsed = _parse_grobid_tei(root)
    elif parser == "jats":
        parsed = _parse_jats(root)
    else:
        parsed = _parse_generic_xml(root)

    caveats = parser_caveats(
        parser,
        body_text=str(parsed.get("body_text", "") or ""),
        abstract=str(parsed.get("abstract", "") or ""),
        sections=parsed.get("sections", []),
        section_records=parsed.get("section_records", []),
        fallback=parser == "generic_xml",
    )
    source_id = stable_id("source", file_meta["source_sha256"])
    document_id = stable_id(
        "doc",
        parsed.get("doi", ""),
        parsed.get("pmcid", ""),
        parsed.get("pmid", ""),
        file_meta["source_sha256"],
    )
    return {
        "source_id": source_id,
        "document_id": document_id,
        "title": parsed.get("title", "") or normalize_text(path.stem.replace("_", " ")),
        "abstract": parsed.get("abstract", ""),
        "text": parsed.get("body_text", ""),
        "doi": parsed.get("doi", ""),
        "source_path": str(path),
        "source_name": path.name,
        "source_format": path.suffix.lower().lstrip(".") or "xml",
        "source_sha256": file_meta["source_sha256"],
        "source_size_bytes": file_meta["source_size_bytes"],
        "source_modified_time_utc": file_meta["source_modified_time_utc"],
        "source_record_count": 1,
        "source_file_status": "ok",
        "ingestion_method": parser,
        "ingestion_schema_version": "literature_ingestion_v1",
        "source_provenance": local_file_provenance(
            file_meta,
            source_type=parser,
            content_scope=_xml_content_scope(parser, parsed, caveats),
            parser_name=parser,
            status="warning" if caveats["warnings"] else "ok",
            warnings=caveats["warnings"],
            limitations=caveats["limitations"],
            next_handoff="litdatamatcher run",
            metadata={"parser_caveats": caveats},
        ).to_dict(),
        "metadata": {
            "xml_parser": parser,
            "xml_root": _local_name(root),
            "xml_sections": parsed.get("sections", []),
            "xml_section_records": parsed.get("section_records", []),
            "xml_parser_caveats": caveats,
            "authors": parsed.get("authors", []),
            "pmcid": parsed.get("pmcid", ""),
            "pmid": parsed.get("pmid", ""),
            "journal": parsed.get("journal", ""),
        },
    }


def detect_xml_parser(root: ET.Element) -> str:
    """Identify the most specific XML article parser available."""

    root_name = _local_name(root).lower()
    namespace = root.tag.split("}", 1)[0].lstrip("{").lower() if "}" in root.tag else ""
    if root_name == "tei" or "tei-c.org" in namespace:
        return "grobid_tei"
    if root_name == "article" or _first(root, "article-meta") is not None:
        return "jats"
    return "generic_xml"


def _parse_jats(root: ET.Element) -> JsonDict:
    """Extract article metadata and section text from JATS/PMC XML."""

    article_meta = _first(root, "article-meta")
    if article_meta is None:
        article_meta = root
    journal_meta = _first(root, "journal-meta")
    title = _text(_first(article_meta, "article-title"))
    abstract = _text(_first(article_meta, "abstract"))
    body = _first(root, "body")
    body_text, sections, section_records = _body_sections(body)
    return {
        "title": title,
        "abstract": abstract,
        "body_text": body_text,
        "doi": _article_id(article_meta, "doi"),
        "pmid": _article_id(article_meta, "pmid"),
        "pmcid": _article_id(article_meta, "pmc") or _article_id(article_meta, "pmcid"),
        "journal": _text(_first(journal_meta, "journal-title")) if journal_meta is not None else "",
        "sections": sections,
        "section_records": section_records,
        "authors": _jats_authors(article_meta),
    }


def _parse_grobid_tei(root: ET.Element) -> JsonDict:
    """Extract bibliographic metadata and body text from GROBID TEI XML."""

    header = _first(root, "teiHeader")
    if header is None:
        header = root
    file_desc = _first(header, "fileDesc")
    if file_desc is None:
        file_desc = header
    title_stmt = _first(file_desc, "titleStmt")
    if title_stmt is None:
        title_stmt = file_desc
    title = _text(_first(title_stmt, "title"))
    abstract = _text(_first(header, "abstract"))
    text = _first(root, "text")
    if text is None:
        text = root
    body = _first(text, "body")
    if body is None:
        body = text
    body_text, sections, section_records = _body_sections(body, section_tag="div", title_tag="head")
    return {
        "title": title,
        "abstract": abstract,
        "body_text": body_text,
        "doi": _tei_idno(root, "doi"),
        "pmid": _tei_idno(root, "pmid"),
        "pmcid": _tei_idno(root, "pmcid"),
        "journal": _text(_first(root, "monogr")),
        "sections": sections,
        "section_records": section_records,
        "authors": _tei_authors(root),
    }


def _parse_generic_xml(root: ET.Element) -> JsonDict:
    """Extract conservative title, abstract, and body text from unknown XML."""

    body = _first(root, "body")
    if body is None:
        body = root
    body_text, sections, section_records = _body_sections(body)
    return {
        "title": _text(_first(root, "title")) or _text(_first(root, "article-title")),
        "abstract": _text(_first(root, "abstract")),
        "body_text": body_text,
        "doi": _text(_first(root, "doi")),
        "pmid": _text(_first(root, "pmid")),
        "pmcid": _text(_first(root, "pmcid")),
        "journal": _text(_first(root, "journal-title")),
        "sections": sections,
        "section_records": section_records,
        "authors": [],
    }


def _body_sections(
    body: ET.Element | None, section_tag: str = "sec", title_tag: str = "title"
) -> tuple[str, list[str], list[JsonDict]]:
    """Return body text, section headings, and reviewable section records."""

    if body is None:
        return "", [], []
    sections: list[str] = []
    section_records: list[JsonDict] = []
    blocks: list[str] = []
    section_elements = [item for item in body.iter() if _local_name(item) == section_tag]
    if not section_elements:
        text = _text(body)
        # Fallback section records preserve review context without claiming section structure.
        return text, sections, [{"title": "", "text": text, "paragraph_count": 1 if text else 0}]
    for section in section_elements:
        heading = _text(_first_child(section, title_tag))
        if heading:
            sections.append(heading)
            blocks.append(heading)
        paragraphs = [_text(item) for item in section.iter() if _local_name(item) in {"p", "head"}]
        paragraphs = [item for item in paragraphs if item and item != heading]
        if paragraphs:
            blocks.extend(paragraphs)
        elif _text(section) and _text(section) != heading:
            blocks.append(_text(section))
            paragraphs = [_text(section)]
        section_text = normalize_text("\n".join(paragraphs))
        section_records.append(
            {
                "title": heading,
                "text": section_text,
                "paragraph_count": len(paragraphs),
            }
        )
    return normalize_text("\n".join(blocks)), _dedupe(sections), section_records


def _article_id(article_meta: ET.Element, pub_id_type: str) -> str:
    """Return a JATS article-id by pub-id-type."""

    for article_id in article_meta.iter():
        if _local_name(article_id) != "article-id":
            continue
        if str(article_id.attrib.get("pub-id-type", "")).lower() == pub_id_type:
            return normalize_text("".join(article_id.itertext()))
    return ""


def _tei_idno(root: ET.Element, id_type: str) -> str:
    """Return a TEI idno value by type."""

    for item in root.iter():
        if _local_name(item) != "idno":
            continue
        if str(item.attrib.get("type", "")).lower() == id_type:
            return normalize_text("".join(item.itertext()))
    return ""


def _jats_authors(article_meta: ET.Element) -> list[str]:
    """Return compact author names from JATS contrib-group metadata."""

    authors: list[str] = []
    for contrib in article_meta.iter():
        if _local_name(contrib) != "contrib":
            continue
        if str(contrib.attrib.get("contrib-type", "")).lower() not in {"", "author"}:
            continue
        name = _first(contrib, "name")
        if name is None:
            collab = _text(_first(contrib, "collab"))
            if collab:
                authors.append(collab)
            continue
        surname = _text(_first(name, "surname"))
        given = _text(_first(name, "given-names"))
        display = " ".join(part for part in [surname, given] if part)
        if display:
            authors.append(display)
    return _dedupe(authors)


def _tei_authors(root: ET.Element) -> list[str]:
    """Return compact author names from GROBID TEI analytic metadata."""

    authors: list[str] = []
    for author in root.iter():
        if _local_name(author) != "author":
            continue
        pers_name = _first(author, "persName")
        if pers_name is None:
            pers_name = author
        surname = _text(_first(pers_name, "surname"))
        forename = _text(_first(pers_name, "forename"))
        display = " ".join(part for part in [surname, forename] if part)
        if display:
            authors.append(display)
    return _dedupe(authors)


def _first(root: ET.Element | None, local_name: str) -> ET.Element | None:
    """Return the first descendant with a local tag name."""

    if root is None:
        return None
    for item in root.iter():
        if _local_name(item) == local_name:
            return item
    return None


def _first_child(root: ET.Element | None, local_name: str) -> ET.Element | None:
    """Return the first direct child with a local tag name."""

    if root is None:
        return None
    for item in list(root):
        if _local_name(item) == local_name:
            return item
    return None


def _local_name(element: ET.Element) -> str:
    """Return an XML tag without namespace decoration."""

    return str(element.tag).rsplit("}", 1)[-1]


def _text(element: ET.Element | None) -> str:
    """Return normalized text from an XML element."""

    if element is None:
        return ""
    return normalize_text(" ".join(part.strip() for part in element.itertext() if part.strip()))


def _dedupe(values: list[str]) -> list[str]:
    """Return nonblank values with order-preserving deduplication."""

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _parser_limitations(parser: str) -> list[str]:
    """Return conservative parser caveats for structured XML provenance."""

    if parser == "grobid_tei":
        return ["TEI quality depends on upstream GROBID PDF parsing."]
    if parser == "jats":
        return ["JATS structure is preserved, but schema validation is not yet performed."]
    return ["Generic XML parsing may miss article-specific metadata."]


def _xml_content_scope(parser: str, parsed: JsonDict, caveats: JsonDict) -> str:
    """Return content scope after parser-quality inspection."""

    if "missing_body_text" in caveats.get("warning_codes", []):
        return "metadata_or_abstract_only"
    if parser == "generic_xml":
        return "partial_xml_text"
    return "structured_full_text"
