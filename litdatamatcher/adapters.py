"""Optional live literature and dataset adapters.

The default pipeline remains offline and deterministic. These adapters provide
the scaffolding needed to extend LitDataMatcher toward live repository scraping
while preserving caching, provenance, and schema normalization.

Adapters retrieve and normalize source metadata; parsers and downstream nodes
decide how much text or dataset detail can be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import xml.etree.ElementTree as ET

from .datasets import classify_dataset_record
from .http_cache import CachedHttpClient
from .provenance import remote_source_provenance, source_profile
from .schemas import DatasetRecord, JsonDict


class LiteratureSourceAdapter(Protocol):
    """Protocol for literature search adapters."""

    name: str

    def search_literature(self, query: str, limit: int = 25) -> list[dict]:
        """Return literature records with title, abstract, doi, and source_id."""


@dataclass(slots=True)
class OpenAlexLiteratureAdapter:
    """OpenAlex scholarly metadata adapter, not an article-body parser."""

    client: CachedHttpClient
    name: str = "openalex"

    def search_literature(self, query: str, limit: int = 25) -> list[dict]:
        """Search OpenAlex works and return normalized literature rows."""

        data = self.client.get_json(
            "https://api.openalex.org/works",
            params={"search": query, "per-page": min(max(1, limit), 200)},
        )
        rows: list[dict] = []
        for work in data.get("results", []):
            abstract = _openalex_abstract(work.get("abstract_inverted_index") or {})
            source_url = work.get("id", "")
            # OpenAlex may expose abstracts, but the adapter still returns discovery metadata.
            provenance = remote_source_provenance(
                source_type="openalex",
                source_url=source_url,
                adapter_name=self.name,
                acquisition_method="openalex_api",
                content_scope="metadata_plus_reconstructed_abstract"
                if abstract
                else "metadata_only",
                raw_record_id=source_url,
                limitations=[
                    "OpenAlex abstracts are reconstructed from an inverted index when available."
                ],
                next_handoff="litdatamatcher run",
                metadata={"source_profile": source_profile("openalex")},
            ).to_dict()
            rows.append(
                {
                    "source_id": work.get("id", ""),
                    "title": work.get("title") or work.get("display_name") or "",
                    "abstract": abstract,
                    "doi": (work.get("doi") or "").replace("https://doi.org/", ""),
                    "year": work.get("publication_year"),
                    "source": self.name,
                    "source_provenance": provenance,
                    "metadata": {
                        "cited_by_count": work.get("cited_by_count", 0),
                        "concepts": [
                            concept.get("display_name", "")
                            for concept in work.get("concepts", [])[:10]
                        ],
                        "source_provenance": provenance,
                    },
                }
            )
        return rows


@dataclass(slots=True)
class PubMedLiteratureAdapter:
    """NCBI PubMed adapter for abstract and citation metadata."""

    client: CachedHttpClient
    name: str = "pubmed"

    def search_literature(self, query: str, limit: int = 25) -> list[dict]:
        """Search PubMed and return normalized literature rows with abstracts when available."""

        ids = _ncbi_search(self.client, "pubmed", query, limit)
        if not ids:
            return []
        data = self.client.get_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
        )
        xml_records = _pubmed_efetch_records(self.client, ids)
        results = data.get("result", {})
        rows: list[dict] = []
        for uid in ids:
            item = results.get(uid, {})
            xml_item = xml_records.get(uid, {})
            article_ids = item.get("articleids", []) if isinstance(item, dict) else []
            doi = xml_item.get("doi", "") or _article_id(article_ids, "doi")
            title = xml_item.get("title", "") or item.get("title", "")
            abstract = xml_item.get("abstract", "")
            # PubMed EFetch here is abstract-oriented unless full text is obtained elsewhere.
            provenance = remote_source_provenance(
                source_type="pubmed",
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                adapter_name=self.name,
                acquisition_method="ncbi_eutilities",
                content_scope="abstract_plus_metadata" if abstract else "metadata_only",
                raw_record_id=uid,
                limitations=[]
                if abstract
                else ["PubMed EFetch did not provide an abstract for this record."],
                next_handoff="litdatamatcher run",
                metadata={"source_profile": source_profile("pubmed")},
            ).to_dict()
            rows.append(
                {
                    "source_id": f"pubmed:{uid}",
                    "document_id": f"pubmed:{uid}",
                    "title": title,
                    "abstract": abstract,
                    "doi": doi,
                    "pmid": uid,
                    "source": self.name,
                    "year": xml_item.get("year") or _pubdate_year(item.get("pubdate", "")),
                    "source_provenance": provenance,
                    "metadata": {
                        "esummary": item,
                        "efetch": xml_item,
                        "journal": xml_item.get("journal", ""),
                        "authors": xml_item.get("authors", []),
                        "source_provenance": provenance,
                    },
                }
            )
        return rows


@dataclass(slots=True)
class ClinicalTrialsDatasetAdapter:
    """ClinicalTrials.gov registry metadata adapter, not patient-level data access."""

    client: CachedHttpClient
    name: str = "clinicaltrials"

    def search(self, query: str) -> list[DatasetRecord]:
        """Search ClinicalTrials.gov and normalize study records as datasets."""

        data = self.client.get_json(
            "https://clinicaltrials.gov/api/v2/studies",
            params={"query.term": query, "pageSize": 25, "format": "json"},
        )
        records: list[DatasetRecord] = []
        for study in data.get("studies", []):
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            description = protocol.get("descriptionModule", {})
            design = protocol.get("designModule", {})
            outcomes = protocol.get("outcomesModule", {})
            title = (
                identification.get("briefTitle")
                or identification.get("officialTitle")
                or identification.get("nctId")
                or "Clinical trial"
            )
            variables = [
                {"name": "treatment", "category": "clinical", "completeness": 0.8},
                {"name": "outcome", "category": "clinical", "completeness": 0.8},
            ]
            if outcomes.get("primaryOutcomes"):
                variables.append({"name": "disease_activity", "category": "phenotype", "completeness": 0.55})
            dataset_id = identification.get("nctId", "")
            source_url = f"https://clinicaltrials.gov/study/{dataset_id}"
            provenance = remote_source_provenance(
                source_type="clinicaltrials",
                source_url=source_url,
                adapter_name=self.name,
                acquisition_method="clinicaltrials_api",
                content_scope="study_metadata",
                raw_record_id=dataset_id,
                limitations=["ClinicalTrials.gov metadata does not guarantee individual-level data access."],
                next_handoff="litdatamatcher run",
                metadata={"source_profile": source_profile("clinicaltrials")},
            ).to_dict()
            records.append(
                classify_dataset_record(
                    {
                        "dataset_id": dataset_id,
                        "title": title,
                        "source": "ClinicalTrials.gov",
                        "description": description.get("briefSummary", ""),
                        "url": source_url,
                        "variables": variables,
                        "populations": ["human"],
                        "assay_types": ["clinical registry"],
                        "sample_size": design.get("enrollmentInfo", {}).get("count", 0),
                        "license": "public domain US government work",
                        "access_type": "public",
                        "quality_score": 0.72,
                        "metadata": _metadata_with_provenance(study, provenance),
                    }
                )
            )
        return records


@dataclass(slots=True)
class GEODatasetAdapter:
    """NCBI GEO summary adapter, not an analysis-ready matrix downloader."""

    client: CachedHttpClient
    name: str = "geo"

    def search(self, query: str) -> list[DatasetRecord]:
        """Search GEO DataSets and normalize study summaries as datasets."""

        ids = _ncbi_search(self.client, "gds", query, limit=25)
        if not ids:
            return []
        data = self.client.get_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "gds", "id": ",".join(ids), "retmode": "json"},
        )
        results = data.get("result", {})
        records: list[DatasetRecord] = []
        for uid in ids:
            item = results.get(uid, {})
            title = item.get("title", "") or item.get("accession", "") or "GEO dataset"
            summary = item.get("summary", "")
            accession = item.get("accession", "") or f"GEO:{uid}"
            source_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={item.get('accession', uid)}"
            provenance = remote_source_provenance(
                source_type="geo",
                source_url=source_url,
                adapter_name=self.name,
                acquisition_method="ncbi_eutilities",
                content_scope="dataset_metadata",
                raw_record_id=accession,
                limitations=["GEO summaries may require follow-up file inspection for variable-level metadata."],
                next_handoff="litdatamatcher run",
                metadata={"source_profile": source_profile("geo")},
            ).to_dict()
            records.append(
                classify_dataset_record(
                    {
                        "dataset_id": accession,
                        "title": title,
                        "source": "GEO",
                        "description": summary,
                        "url": source_url,
                        "assay_types": [_first_text(item, "gdstype", "entrytype")],
                        "sample_size": _first_int(item, "n_samples", "samples", "sample_count"),
                        "license": "public repository terms",
                        "access_type": "public metadata with study-specific files",
                        "quality_score": 0.68,
                        "metadata": _metadata_with_provenance(item, provenance),
                    }
                )
            )
        return records


@dataclass(slots=True)
class MGnifyDatasetAdapter:
    """MGnify study metadata adapter, not a sample-table materializer."""

    client: CachedHttpClient
    name: str = "mgnify"

    def search(self, query: str) -> list[DatasetRecord]:
        """Search MGnify studies and normalize v2 or legacy JSON records as datasets."""

        data = self.client.get_json(
            "https://www.ebi.ac.uk/metagenomics/api/v2/studies",
            params={"search": query, "page_size": 25},
        )
        records: list[DatasetRecord] = []
        for item in _mgnify_items(data):
            accession = _first_text(item, "accession", "id", "study_accession")
            title = _first_text(
                item,
                "study_name",
                "study-title",
                "study_title",
                "name",
                "title",
                default=accession or "MGnify study",
            )
            description = _first_text(item, "abstract", "description", "study-abstract")
            sample_size = _mgnify_sample_count(item)
            source_url = _first_text(
                item,
                "url",
                default=f"https://www.ebi.ac.uk/metagenomics/studies/{accession}",
            )
            provenance = remote_source_provenance(
                source_type="mgnify",
                source_url=source_url,
                adapter_name=self.name,
                acquisition_method="mgnify_api_v2",
                content_scope="dataset_metadata",
                raw_record_id=accession,
                limitations=["MGnify list metadata may need study detail follow-up for full sample context."],
                next_handoff="litdatamatcher run",
                metadata={"source_profile": source_profile("mgnify")},
            ).to_dict()
            records.append(
                classify_dataset_record(
                    {
                        "dataset_id": accession,
                        "title": title,
                        "source": "MGnify",
                        "description": description,
                        "url": source_url,
                        "variables": [
                            {
                                "name": "microbiome_composition",
                                "category": "omics",
                                "observed_count": sample_size,
                                "completeness": 0.8,
                            },
                            {
                                "name": "body_site",
                                "category": "metadata",
                                "observed_count": 0,
                                "completeness": 0.45,
                            },
                        ],
                        "populations": [],
                        "organisms": ["microbiome"],
                        "assay_types": [_first_text(item, "experiment-type", "experiment_type")],
                        "sample_size": sample_size,
                        "license": "EMBL-EBI public metadata terms",
                        "access_type": "public metadata with study-specific files",
                        "quality_score": 0.72,
                        "metadata": _metadata_with_provenance(item, provenance),
                    }
                )
            )
        return records


DATASET_ADAPTERS = {
    "clinicaltrials": ClinicalTrialsDatasetAdapter,
    "geo": GEODatasetAdapter,
    "mgnify": MGnifyDatasetAdapter,
}

LITERATURE_ADAPTERS = {
    "openalex": OpenAlexLiteratureAdapter,
    "pubmed": PubMedLiteratureAdapter,
}


def adapter_source_profiles() -> JsonDict:
    """Return source profiles for optional literature and dataset adapters."""

    return {
        "literature": {
            name: source_profile(name) for name in sorted(LITERATURE_ADAPTERS)
        },
        "datasets": {
            name: source_profile(name) for name in sorted(DATASET_ADAPTERS)
        },
    }


def build_dataset_adapters(
    names: list[str] | tuple[str, ...],
    client: CachedHttpClient | None = None,
) -> list[object]:
    """Build named live dataset adapters with a shared cached HTTP client."""

    client = client or CachedHttpClient()
    adapters: list[object] = []
    for name in names:
        normalized = str(name or "").strip().lower()
        if normalized not in DATASET_ADAPTERS:
            raise ValueError(f"Unknown dataset adapter {name!r}; choose {sorted(DATASET_ADAPTERS)}")
        adapters.append(DATASET_ADAPTERS[normalized](client))
    return adapters


def build_literature_adapters(
    names: list[str] | tuple[str, ...],
    client: CachedHttpClient | None = None,
) -> list[LiteratureSourceAdapter]:
    """Build named live literature adapters with a shared cached HTTP client."""

    client = client or CachedHttpClient()
    adapters: list[LiteratureSourceAdapter] = []
    for name in names:
        normalized = str(name or "").strip().lower()
        if normalized not in LITERATURE_ADAPTERS:
            raise ValueError(
                f"Unknown literature adapter {name!r}; choose {sorted(LITERATURE_ADAPTERS)}"
            )
        adapters.append(LITERATURE_ADAPTERS[normalized](client))
    return adapters


def search_dataset_sources(
    query: str,
    source_names: list[str] | tuple[str, ...],
    client: CachedHttpClient | None = None,
    limit: int = 25,
) -> list[DatasetRecord]:
    """Search named live dataset sources and deduplicate normalized records."""

    records: list[DatasetRecord] = []
    seen: set[str] = set()
    for adapter in build_dataset_adapters(source_names, client=client):
        for record in adapter.search(query):
            if record.dataset_id in seen:
                continue
            seen.add(record.dataset_id)
            records.append(record)
            if len(records) >= limit:
                return records
    return records


def search_literature_sources(
    query: str,
    source_names: list[str] | tuple[str, ...],
    client: CachedHttpClient | None = None,
    limit: int = 25,
) -> list[JsonDict]:
    """Search named live literature metadata sources."""

    rows: list[JsonDict] = []
    seen: set[str] = set()
    for adapter in build_literature_adapters(source_names, client=client):
        for row in adapter.search_literature(query, limit=limit):
            key = str(row.get("source_id", "") or row.get("doi", "") or row.get("title", ""))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows


def cached_client(cache_dir: str | Path | None = None) -> CachedHttpClient:
    """Return a cached HTTP client for CLI adapter commands."""

    return CachedHttpClient(cache_dir=cache_dir or Path("local/http_cache"))


def _openalex_abstract(index: dict[str, list[int]]) -> str:
    """Reconstruct an OpenAlex inverted-index abstract."""

    if not index:
        return ""
    tokens: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            tokens.append((int(position), word))
    return " ".join(word for _, word in sorted(tokens))


def _ncbi_search(client: CachedHttpClient, database: str, query: str, limit: int) -> list[str]:
    """Return NCBI identifiers from an ESearch JSON response."""

    data = client.get_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": database,
            "term": query,
            "retmode": "json",
            "retmax": min(max(1, limit), 200),
        },
    )
    return [str(uid) for uid in data.get("esearchresult", {}).get("idlist", [])]


def _article_id(article_ids: list[JsonDict], id_type: str) -> str:
    """Return a PubMed article identifier by type."""

    for item in article_ids:
        if str(item.get("idtype", "")).lower() == id_type:
            return str(item.get("value", "") or "")
    return ""


def _metadata_with_provenance(raw_metadata: JsonDict, provenance: JsonDict) -> JsonDict:
    """Attach adapter provenance while preserving the raw API metadata shape."""

    metadata = dict(raw_metadata or {})
    metadata["source_provenance"] = provenance
    return metadata


def _pubmed_efetch_records(client: CachedHttpClient, ids: list[str]) -> dict[str, JsonDict]:
    """Fetch and parse PubMed XML records, falling back to empty metadata on failure."""

    try:
        xml_text = client.get_text(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        )
    except Exception:
        return {}
    return _parse_pubmed_xml(xml_text)


def _parse_pubmed_xml(xml_text: str) -> dict[str, JsonDict]:
    """Parse the subset of PubMed XML needed for literature rows."""

    try:
        root = ET.fromstring(xml_text.encode("utf-8"))
    except ET.ParseError:
        return {}
    records: dict[str, JsonDict] = {}
    for article in root.iter():
        if _xml_name(article) != "PubmedArticle":
            continue
        medline = _first_xml(article, "MedlineCitation")
        article_node = _first_xml(medline, "Article")
        pmid = _xml_text(_first_xml(medline, "PMID"))
        if not pmid:
            continue
        records[pmid] = {
            "pmid": pmid,
            "title": _xml_text(_first_xml(article_node, "ArticleTitle")),
            "abstract": _pubmed_abstract(article_node),
            "doi": _pubmed_article_id(article, "doi"),
            "pmcid": _pubmed_article_id(article, "pmc"),
            "journal": _xml_text(_first_xml(_first_xml(article_node, "Journal"), "Title")),
            "year": _pubmed_xml_year(article_node),
            "authors": _pubmed_authors(article_node),
        }
    return records


def _pubmed_abstract(article_node: ET.Element | None) -> str:
    """Return a joined abstract from labeled or unlabeled PubMed abstract parts."""

    if article_node is None:
        return ""
    parts: list[str] = []
    for item in article_node.iter():
        if _xml_name(item) != "AbstractText":
            continue
        label = str(item.attrib.get("Label", "") or "").strip()
        text = _xml_text(item)
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts)


def _pubmed_article_id(article: ET.Element, id_type: str) -> str:
    """Return one PubMed ArticleId value by IdType."""

    target = id_type.lower()
    for item in article.iter():
        if _xml_name(item) == "ArticleId" and str(item.attrib.get("IdType", "")).lower() == target:
            return _xml_text(item)
    return ""


def _pubmed_xml_year(article_node: ET.Element | None) -> int | None:
    """Extract the best available publication year from PubMed XML."""

    journal_issue = _first_xml(_first_xml(article_node, "Journal"), "JournalIssue")
    pub_date = _first_xml(journal_issue, "PubDate")
    year = _xml_text(_first_xml(pub_date, "Year"))
    if year.isdigit() and len(year) == 4:
        return int(year)
    medline_date = _xml_text(_first_xml(pub_date, "MedlineDate"))
    return _pubdate_year(medline_date)


def _pubmed_authors(article_node: ET.Element | None) -> list[str]:
    """Return compact author display names from a PubMed article node."""

    if article_node is None:
        return []
    authors: list[str] = []
    author_list = _first_xml(article_node, "AuthorList")
    for author in list(author_list) if author_list is not None else []:
        if _xml_name(author) != "Author":
            continue
        collective = _xml_text(_first_xml(author, "CollectiveName"))
        if collective:
            authors.append(collective)
            continue
        last = _xml_text(_first_xml(author, "LastName"))
        initials = _xml_text(_first_xml(author, "Initials"))
        name = " ".join(part for part in [last, initials] if part)
        if name:
            authors.append(name)
    return authors


def _first_xml(root: ET.Element | None, local_name: str) -> ET.Element | None:
    """Return the first descendant with a local XML tag name."""

    if root is None:
        return None
    for item in root.iter():
        if _xml_name(item) == local_name:
            return item
    return None


def _xml_name(element: ET.Element) -> str:
    """Return an XML tag name without namespace decoration."""

    return str(element.tag).rsplit("}", 1)[-1]


def _xml_text(element: ET.Element | None) -> str:
    """Return normalized text from an XML element."""

    if element is None:
        return ""
    return " ".join(part.strip() for part in element.itertext() if part.strip())


def _mgnify_items(data: JsonDict) -> list[JsonDict]:
    """Normalize MGnify v2 list rows and legacy JSON:API rows into flat dicts."""

    if isinstance(data.get("items"), list):
        return [item for item in data["items"] if isinstance(item, dict)]
    legacy_rows: list[JsonDict] = []
    for item in data.get("data", []):
        if not isinstance(item, dict):
            continue
        attrs = dict(item.get("attributes", {}) or {})
        attrs.setdefault("id", item.get("id", ""))
        attrs.setdefault("accession", item.get("id", ""))
        if item.get("relationships"):
            attrs["relationships"] = item.get("relationships")
        legacy_rows.append(attrs)
    return legacy_rows


def _mgnify_sample_count(item: JsonDict) -> int:
    """Return sample count from common MGnify v2 and legacy field shapes."""

    direct = _first_int(
        item,
        "sample_count",
        "samples_count",
        "sample-count",
        "samples-count",
        "num_samples",
    )
    if direct:
        return direct
    relationships = item.get("relationships", {})
    if isinstance(relationships, dict):
        samples = relationships.get("samples", {})
        if isinstance(samples, dict):
            meta = samples.get("meta", {})
            if isinstance(meta, dict):
                return _first_int(meta, "count", "total")
    return 0


def _pubdate_year(pubdate: str) -> int | None:
    """Extract a publication year from a PubMed date string."""

    for token in str(pubdate or "").split():
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def _first_text(payload: JsonDict, *keys: str, default: str = "") -> str:
    """Return the first nonblank string from possible API field names."""

    for key in keys:
        value = payload.get(key, "")
        if str(value or "").strip():
            return str(value).strip()
    return default


def _first_int(payload: JsonDict, *keys: str) -> int:
    """Return the first parseable integer from possible API field names."""

    for key in keys:
        value = payload.get(key, "")
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0
