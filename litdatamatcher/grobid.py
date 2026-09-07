"""Optional GROBID client for converting PDFs into TEI XML files.

This bridge calls a separately running GROBID service and writes TEI XML; normal
ingestion can still consume existing TEI files without running GROBID.
"""

from __future__ import annotations

from pathlib import Path

from .http_cache import CachedHttpClient
from .schemas import JsonDict


DEFAULT_GROBID_URL = "http://localhost:8070"
FULLTEXT_ENDPOINT = "/api/processFulltextDocument"


def process_pdf_to_tei(
    pdf_path: str | Path,
    out_path: str | Path,
    *,
    server_url: str = DEFAULT_GROBID_URL,
    consolidate_header: bool = False,
    consolidate_citations: bool = False,
    include_raw_affiliations: bool = False,
    client: CachedHttpClient | None = None,
) -> JsonDict:
    """Convert one PDF to GROBID TEI XML through a running GROBID service."""

    pdf_path = Path(pdf_path)
    out_path = Path(out_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"GROBID input should be a PDF file: {pdf_path}")

    client = client or CachedHttpClient(cache_dir=Path("local/http_cache/grobid"))
    endpoint = _endpoint_url(server_url)
    form_data = {
        "consolidateHeader": int(bool(consolidate_header)),
        "consolidateCitations": int(bool(consolidate_citations)),
        "includeRawAffiliations": int(bool(include_raw_affiliations)),
    }
    tei_xml = client.post_file_text(endpoint, pdf_path, field_name="input", data=form_data)
    # The output is an intermediate TEI artifact; ingestion creates literature records later.
    if "<TEI" not in tei_xml and "<tei:" not in tei_xml:
        raise RuntimeError("GROBID response did not look like TEI XML.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tei_xml, encoding="utf-8")
    return {
        "input": str(pdf_path),
        "out": str(out_path),
        "server_url": server_url.rstrip("/"),
        "endpoint": endpoint,
        "bytes_written": out_path.stat().st_size,
        "ingestion_next_step": f"litdatamatcher ingest --input {out_path} --out run/corpus/literature.jsonl",
    }


def _endpoint_url(server_url: str) -> str:
    """Return the GROBID full-text endpoint for a service base URL."""

    return server_url.rstrip("/") + FULLTEXT_ENDPOINT
