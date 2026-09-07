"""Optional live literature and dataset adapters.

The default pipeline remains offline and deterministic. These adapters provide
the scaffolding needed to extend LitDataMatcher toward live repository scraping
while preserving caching, provenance, and schema normalization.

Adapters retrieve and normalize source metadata; parsers and downstream nodes
decide how much text or dataset detail can be used.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
class EuropePMCLiteratureAdapter:
    """Europe PMC metadata adapter with stable source and publication identifiers."""

    client: CachedHttpClient
    name: str = "europepmc"

    def search_literature(self, query: str, limit: int = 25) -> list[dict]:
        """Search Europe PMC core records without requesting article bodies."""

        data = self.client.get_json(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "resultType": "core", "pageSize": min(max(1, limit), 100)},
        )
        response_metadata = _client_response_metadata(self.client)
        rows: list[dict] = []
        for item in data.get("resultList", {}).get("result", []):
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "") or "").strip().upper()
            record_id = str(item.get("id", "") or "").strip()
            title = str(item.get("title", "") or "").strip()
            if not source or not record_id or not title:
                continue
            doi = _normalize_doi(item.get("doi", ""))
            stable_id = f"europepmc:{source}:{record_id}"
            source_url = f"https://europepmc.org/article/{source}/{record_id}"
            abstract = str(item.get("abstractText", "") or "").strip()
            version_relationships = item.get("commentCorrectionList", {})
            provenance = remote_source_provenance(
                source_type="europepmc",
                source_url=source_url,
                adapter_name=self.name,
                adapter_version="europepmc_rest_search_v1",
                retrieval_time_utc=str(response_metadata.get("retrieval_time_utc", "")),
                acquisition_method="europepmc_rest_api",
                content_scope="abstract_plus_metadata" if abstract else "metadata_only",
                raw_record_id=stable_id,
                limitations=["Europe PMC search records are metadata; no article body was requested."],
                next_handoff="litdatamatcher run",
                metadata={
                    "source_profile": source_profile("europepmc"),
                    "source_database": source,
                    "cache_snapshot": response_metadata,
                },
            ).to_dict()
            rows.append(
                {
                    "source_id": stable_id,
                    "document_id": stable_id,
                    "title": title,
                    "abstract": abstract,
                    "doi": doi,
                    "pmid": record_id if source == "MED" else "",
                    "pmcid": str(item.get("pmcid", "") or ""),
                    "year": _year_from_value(item.get("firstPublicationDate", "") or item.get("pubYear", "")),
                    "source": self.name,
                    "version_relationships": version_relationships if isinstance(version_relationships, dict) else {},
                    "source_provenance": provenance,
                    "metadata": {
                        "europepmc_source": source,
                        "journal": str(item.get("journalTitle", "") or ""),
                        "first_publication_date": str(item.get("firstPublicationDate", "") or ""),
                        "version_relationships": version_relationships if isinstance(version_relationships, dict) else {},
                        "source_provenance": provenance,
                    },
                }
            )
        return rows


@dataclass(slots=True)
class CrossrefLiteratureAdapter:
    """Crossref DOI metadata adapter with update and relation provenance."""

    client: CachedHttpClient
    name: str = "crossref"

    def search_literature(self, query: str, limit: int = 25) -> list[dict]:
        """Search Crossref works and normalize only DOI-addressable records."""

        data = self.client.get_json(
            "https://api.crossref.org/works",
            params={"query": query, "rows": min(max(1, limit), 100), "select": "DOI,title,abstract,published,published-online,published-print,indexed,created,relation,update-policy,container-title,author,type"},
        )
        response_metadata = _client_response_metadata(self.client)
        rows: list[dict] = []
        for item in data.get("message", {}).get("items", []):
            if not isinstance(item, dict):
                continue
            doi = _normalize_doi(item.get("DOI", ""))
            title = _first_sequence_text(item.get("title", []))
            if not doi or not title:
                continue
            source_id = f"crossref:{doi}"
            source_url = f"https://doi.org/{doi}"
            version_relationships = item.get("relation", {})
            provenance = remote_source_provenance(
                source_type="crossref",
                source_url=source_url,
                adapter_name=self.name,
                adapter_version="crossref_works_v1",
                retrieval_time_utc=str(response_metadata.get("retrieval_time_utc", "")),
                acquisition_method="crossref_works_api",
                content_scope="metadata_only",
                raw_record_id=source_id,
                limitations=["Crossref records are DOI metadata and are not article-body evidence."],
                next_handoff="litdatamatcher run",
                metadata={
                    "source_profile": source_profile("crossref"),
                    "indexed": item.get("indexed", {}),
                    "created": item.get("created", {}),
                    "cache_snapshot": response_metadata,
                },
            ).to_dict()
            rows.append(
                {
                    "source_id": source_id,
                    "document_id": source_id,
                    "title": title,
                    "abstract": _strip_jats(str(item.get("abstract", "") or "")),
                    "doi": doi,
                    "year": _crossref_year(item),
                    "source": self.name,
                    "version_relationships": version_relationships if isinstance(version_relationships, dict) else {},
                    "source_provenance": provenance,
                    "metadata": {
                        "container_title": _first_sequence_text(item.get("container-title", [])),
                        "authors": _crossref_authors(item.get("author", [])),
                        "type": str(item.get("type", "") or ""),
                        "indexed": item.get("indexed", {}),
                        "created": item.get("created", {}),
                        "update_policy": str(item.get("update-policy", "") or ""),
                        "version_relationships": version_relationships if isinstance(version_relationships, dict) else {},
                        "source_provenance": provenance,
                    },
                }
            )
        return rows


@dataclass(slots=True)
class ClinicalTrialsDatasetAdapter:
    """ClinicalTrials.gov study-metadata adapter, never a participant-data downloader."""

    client: CachedHttpClient
    name: str = "clinicaltrials"

    def search(self, query: str) -> list[DatasetRecord]:
        """Search and normalize bounded study metadata with explicit design caveats."""

        data = self.client.get_json(
            "https://clinicaltrials.gov/api/v2/studies",
            params={"query.term": query, "pageSize": 25, "format": "json"},
        )
        response_metadata = _client_response_metadata(self.client)
        records_by_id: dict[str, DatasetRecord] = {}
        for study in data.get("studies", []):
            if not isinstance(study, dict):
                continue
            record = _clinicaltrials_record(study, response_metadata=response_metadata)
            if record is None:
                continue
            existing = records_by_id.get(record.dataset_id)
            if existing is None or _clinicaltrials_version(record) > _clinicaltrials_version(existing):
                if existing is not None:
                    _append_clinical_version(record, existing)
                records_by_id[record.dataset_id] = record
            else:
                _append_clinical_version(existing, record)
        return list(records_by_id.values())


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


@dataclass(slots=True)
class ENASRADatasetAdapter:
    """ENA/SRA run-index adapter that groups technical runs under a study record."""

    client: CachedHttpClient
    name: str = "ena"

    def search(self, query: str) -> list[DatasetRecord]:
        """Search a bounded ENA read-run page and group rows by stable study accession."""

        data = self.client.get_json(
            "https://www.ebi.ac.uk/ena/portal/api/search",
            params={
                "result": "read_run",
                "query": query,
                "fields": "study_accession,secondary_study_accession,secondary_project,study_title,study_alias,run_accession,experiment_accession,sample_accession,secondary_sample_accession,sample_alias,sample_title,sample_description,scientific_name,library_strategy,library_source,library_selection,fastq_ftp,submitted_ftp,sra_ftp,first_public,last_updated",
                "format": "json",
                "limit": 100,
            },
        )
        response_metadata = _client_response_metadata(self.client)
        rows = data if isinstance(data, list) else data.get("data", [])
        grouped: dict[str, list[JsonDict]] = {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            study_id = str(row.get("study_accession", "") or "").strip().upper()
            if not re.fullmatch(r"(?:ERP|SRP|DRP|PRJ(?:EB|NA|DB))\d+", study_id):
                continue
            grouped.setdefault(study_id, []).append(row)
        records = [
            _ena_study_record(study_id, study_rows, response_metadata)
            for study_id, study_rows in grouped.items()
        ]
        return [record for record in records if record is not None]


def _ena_study_record(
    study_id: str, rows: list[JsonDict], response_metadata: JsonDict
) -> DatasetRecord | None:
    """Build one study-level record while retaining sample/run distinction in metadata."""

    if not rows:
        return None
    newest = max(rows, key=lambda row: str(row.get("last_updated", "") or ""))
    title = _first_text(newest, "study_title", "study_alias", default=study_id)
    run_ids = _unique_field(rows, "run_accession")
    sample_ids = _unique_field(rows, "sample_accession")
    experiments = _unique_field(rows, "experiment_accession")
    strategies = _unique_field(rows, "library_strategy")
    organisms = _unique_field(rows, "scientific_name")
    samples = [
        {
            "sample_accession": str(row.get("sample_accession", "") or ""),
            "secondary_sample_accession": str(row.get("secondary_sample_accession", "") or ""),
            "sample_alias": str(row.get("sample_alias", "") or ""),
            "attributes": {
                "title": str(row.get("sample_title", "") or ""),
                "description": str(row.get("sample_description", "") or ""),
            },
        }
        for row in _dedupe_rows(rows, "sample_accession")
    ]
    provenance = remote_source_provenance(
        source_type="ena",
        source_url=f"https://www.ebi.ac.uk/ena/browser/view/{study_id}",
        adapter_name="ena",
        adapter_version="ena_portal_read_run_v1",
        retrieval_time_utc=str(response_metadata.get("retrieval_time_utc", "")),
        acquisition_method="ena_portal_api",
        content_scope="study_sample_run_metadata",
        raw_record_id=study_id,
        limitations=[
            "ENA run metadata does not establish donor identity or biological-sample independence.",
            "Run accessions are technical records and are not equated with biological samples.",
        ],
        next_handoff="dataset matching",
        metadata={"source_profile": source_profile("ena"), "cache_snapshot": response_metadata},
    ).to_dict()
    raw_available = any(_has_remote_file(row, "fastq_ftp", "sra_ftp") for row in rows)
    processed_available = any(_has_remote_file(row, "submitted_ftp") for row in rows)
    metadata = {
        "secondary_study_accessions": _unique_field(rows, "secondary_study_accession"),
        "bioproject_accessions": _unique_field(rows, "secondary_project"),
        "runs": run_ids,
        "experiments": experiments,
        "samples": samples,
        "run_sample_links": [
            {"run_accession": str(row.get("run_accession", "") or ""), "sample_accession": str(row.get("sample_accession", "") or ""), "relation": "TECHNICAL_RUN_OF_DECLARED_SAMPLE"}
            for row in _dedupe_rows(rows, "run_accession")
        ],
        "access_availability": {"raw_reads": raw_available, "submitted_files": processed_available, "interpretation": "availability metadata only; file-level access and processing status require inspection"},
        "version_time": str(newest.get("last_updated", "UNKNOWN") or "UNKNOWN"),
        "pagination": {"page_size": 100, "returned_run_rows": len(rows), "status": "BOUNDED_PAGE_NOT_COMPLETE_CENSUS"},
        "dependence": {"biological_sample_count": len(sample_ids), "technical_run_count": len(run_ids), "donor_links": "AMBIGUOUS_NOT_INFERRED", "deduplication_key": "study_accession+run_accession"},
        "missingness": {"sample_accessions": "PRESENT" if sample_ids else "MISSING", "run_accessions": "PRESENT" if run_ids else "MISSING", "organism": "PRESENT" if organisms else "MISSING", "library_strategy": "PRESENT" if strategies else "MISSING"},
        "source_provenance": provenance,
    }
    variables = [
        {"name": "sequencing_run", "category": "technical_metadata", "observed_count": 0, "completeness": 0.8 if run_ids else 0.0},
        {"name": "biological_sample", "category": "metadata", "observed_count": 0, "completeness": 0.8 if sample_ids else 0.0},
        {"name": "library_strategy", "category": "assay_metadata", "observed_count": 0, "completeness": 0.8 if strategies else 0.0},
    ]
    return classify_dataset_record({"dataset_id": study_id, "title": title, "source": "ENA/SRA", "description": _first_text(newest, "sample_description"), "url": f"https://www.ebi.ac.uk/ena/browser/view/{study_id}", "variables": variables, "populations": [], "organisms": organisms, "assay_types": strategies, "sample_size": 0, "license": "ENA/SRA metadata; source-specific data reuse review required", "access_type": "public metadata; per-file availability as recorded", "quality_score": 0.68, "metadata": metadata})


def _unique_field(rows: list[JsonDict], name: str) -> list[str]:
    return _string_list([row.get(name, "") for row in rows])


def _dedupe_rows(rows: list[JsonDict], key: str) -> list[JsonDict]:
    selected: dict[str, JsonDict] = {}
    for row in rows:
        value = str(row.get(key, "") or "").strip()
        if value and value not in selected:
            selected[value] = row
    return list(selected.values())


def _has_remote_file(row: JsonDict, *names: str) -> bool:
    return any(str(row.get(name, "") or "").strip() for name in names)


def _clinicaltrials_record(
    study: JsonDict, *, response_metadata: JsonDict
) -> DatasetRecord | None:
    """Normalize one ClinicalTrials.gov v2 study without implying data access."""

    protocol = study.get("protocolSection", {})
    if not isinstance(protocol, dict):
        return None
    identification = _clinical_module(protocol, "identificationModule")
    dataset_id = str(identification.get("nctId", "") or "").strip().upper()
    if not re.fullmatch(r"NCT\d{8}", dataset_id):
        return None
    title = _first_text(identification, "briefTitle", "officialTitle", default=dataset_id)
    status = _clinical_module(protocol, "statusModule")
    design = _clinical_module(protocol, "designModule")
    conditions = _string_list(_clinical_module(protocol, "conditionsModule").get("conditions", []))
    arms_module = _clinical_module(protocol, "armsInterventionsModule")
    outcomes_module = _clinical_module(protocol, "outcomesModule")
    eligibility = _clinical_module(protocol, "eligibilityModule")
    study_type = str(design.get("studyType", "UNKNOWN") or "UNKNOWN").upper()
    interventions = _clinical_interventions(arms_module.get("interventions", []))
    arms = _clinical_arms(arms_module.get("armGroups", []))
    primary_outcomes = _clinical_outcomes(outcomes_module.get("primaryOutcomes", []))
    secondary_outcomes = _clinical_outcomes(outcomes_module.get("secondaryOutcomes", []))
    other_outcomes = _clinical_outcomes(outcomes_module.get("otherOutcomes", []))
    enrollment = design.get("enrollmentInfo", {})
    enrollment = enrollment if isinstance(enrollment, dict) else {}
    version_time = _clinical_date(status, "lastUpdatePostDateStruct", "lastUpdateSubmitDate", "studyFirstPostDateStruct")
    source_url = f"https://clinicaltrials.gov/study/{dataset_id}"
    observational = study_type == "OBSERVATIONAL"
    variables = _clinical_variables(
        conditions=conditions,
        interventions=interventions,
        arms=arms,
        primary_outcomes=primary_outcomes,
        secondary_outcomes=[*secondary_outcomes, *other_outcomes],
        eligibility=eligibility,
        study_type=study_type,
    )
    limitations = [
        "ClinicalTrials.gov metadata does not establish individual-level data access or analysis readiness.",
        "Enrollment is registry metadata and is not treated as an analyzed sample count.",
    ]
    if observational:
        limitations.append("Observational study metadata is not interpreted as perturbational evidence.")
    provenance = remote_source_provenance(
        source_type="clinicaltrials",
        source_url=source_url,
        adapter_name="clinicaltrials",
        adapter_version="clinicaltrials_v2_studies",
        retrieval_time_utc=str(response_metadata.get("retrieval_time_utc", "")),
        acquisition_method="clinicaltrials_v2_api",
        content_scope="study_registry_metadata",
        raw_record_id=dataset_id,
        limitations=limitations,
        next_handoff="dataset matching",
        metadata={
            "source_profile": source_profile("clinicaltrials"),
            "cache_snapshot": response_metadata,
            "study_version_time": version_time,
        },
    ).to_dict()
    metadata = _metadata_with_provenance(study, provenance)
    metadata.update(
        {
            "study_type": study_type,
            "overall_status": str(status.get("overallStatus", "UNKNOWN") or "UNKNOWN"),
            "study_version_time": version_time,
            "conditions": conditions,
            "interventions": interventions,
            "comparators": [arm for arm in arms if "COMPARATOR" in str(arm.get("type", ""))],
            "arms_groups": arms,
            "primary_outcomes": primary_outcomes,
            "secondary_outcomes": secondary_outcomes,
            "other_outcomes": other_outcomes,
            "eligibility_population": {
                "sex": str(eligibility.get("sex", "UNKNOWN") or "UNKNOWN"),
                "minimum_age": str(eligibility.get("minimumAge", "UNKNOWN") or "UNKNOWN"),
                "maximum_age": str(eligibility.get("maximumAge", "UNKNOWN") or "UNKNOWN"),
                "healthy_volunteers": eligibility.get("healthyVolunteers", "UNKNOWN"),
                "criteria": str(eligibility.get("eligibilityCriteria", "") or ""),
            },
            "phase": _string_list(design.get("phases", [])),
            "enrollment": {
                "count": _optional_nonnegative_int(enrollment.get("count")),
                "unit": "participants",
                "type": str(enrollment.get("type", "UNKNOWN") or "UNKNOWN"),
                "interpretation": "registry enrollment; not analyzed sample count",
            },
            "design": {
                "allocation": _nested_text(design, "designInfo", "allocation"),
                "intervention_model": _nested_text(design, "designInfo", "interventionModel"),
                "observational_model": _nested_text(design, "designInfo", "observationalModel"),
                "time_perspective": _nested_text(design, "designInfo", "timePerspective"),
            },
            "causal_interpretation": "NOT_PERTURBATIONAL" if observational else "REGISTRY_METADATA_ONLY",
            "access_status": "PUBLIC_REGISTRY_METADATA",
            "missingness": _clinical_missingness(
                conditions, interventions, primary_outcomes, eligibility, enrollment
            ),
        }
    )
    return classify_dataset_record(
        {
            "dataset_id": dataset_id,
            "title": title,
            "source": "ClinicalTrials.gov",
            "description": _first_text(_clinical_module(protocol, "descriptionModule"), "briefSummary", "detailedDescription"),
            "url": source_url,
            "variables": variables,
            "populations": _clinical_population(eligibility),
            "organisms": ["human"],
            "assay_types": ["clinical study registry metadata"],
            "sample_size": 0,
            "license": "public registry metadata; source-specific reuse review required",
            "access_type": "public registry metadata; participant data availability unknown",
            "quality_score": 0.7,
            "metadata": metadata,
        }
    )


def _clinical_module(protocol: JsonDict, name: str) -> JsonDict:
    value = protocol.get(name, {})
    return dict(value) if isinstance(value, dict) else {}


def _clinical_interventions(value: object) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if name:
            rows.append({"name": name, "type": str(item.get("type", "UNKNOWN") or "UNKNOWN"), "description": str(item.get("description", "") or "")})
    return rows


def _clinical_arms(value: object) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "") or "").strip()
        if label:
            rows.append({"label": label, "type": str(item.get("type", "UNKNOWN") or "UNKNOWN"), "description": str(item.get("description", "") or ""), "intervention_names": _string_list(item.get("interventionNames", []))})
    return rows


def _clinical_outcomes(value: object) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        measure = str(item.get("measure", "") or "").strip()
        if measure:
            rows.append({"measure": measure, "time_frame": str(item.get("timeFrame", "UNKNOWN") or "UNKNOWN"), "description": str(item.get("description", "") or "")})
    return rows


def _clinical_variables(**fields: object) -> list[JsonDict]:
    variables: list[JsonDict] = []
    for name, category, values in (
        ("condition", "phenotype", fields["conditions"]),
        ("intervention", "clinical", fields["interventions"]),
        ("study_arm", "study_design_feature", fields["arms"]),
        ("outcome", "clinical", [*fields["primary_outcomes"], *fields["secondary_outcomes"]]),
        ("eligibility_criteria", "metadata", [fields["eligibility"]]),
    ):
        variables.append({"name": name, "category": category, "observed_count": 0, "completeness": 0.8 if values else 0.0})
    variables.append({"name": "study_type", "category": "study_design_feature", "observed_count": 0, "completeness": 1.0})
    if any(item.get("time_frame") not in {"", "UNKNOWN"} for item in [*fields["primary_outcomes"], *fields["secondary_outcomes"]]):
        variables.append({"name": "timepoint", "category": "temporal_structure", "observed_count": 0, "completeness": 0.65})
    return variables


def _clinical_date(status: JsonDict, *names: str) -> str:
    for name in names:
        value = status.get(name, {})
        if isinstance(value, dict) and str(value.get("date", "") or "").strip():
            return str(value["date"])
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "UNKNOWN"


def _nested_text(payload: JsonDict, module: str, name: str) -> str:
    nested = payload.get(module, {})
    return str(nested.get(name, "UNKNOWN") or "UNKNOWN") if isinstance(nested, dict) else "UNKNOWN"


def _optional_nonnegative_int(value: object) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _clinical_population(eligibility: JsonDict) -> list[str]:
    return ["human", f"sex:{eligibility.get('sex', 'UNKNOWN') or 'UNKNOWN'}", f"age:{eligibility.get('minimumAge', 'UNKNOWN') or 'UNKNOWN'} to {eligibility.get('maximumAge', 'UNKNOWN') or 'UNKNOWN'}"]


def _clinical_missingness(conditions: list[str], interventions: list[JsonDict], outcomes: list[JsonDict], eligibility: JsonDict, enrollment: JsonDict) -> JsonDict:
    return {"conditions": "PRESENT" if conditions else "MISSING", "interventions": "PRESENT" if interventions else "MISSING", "outcomes": "PRESENT" if outcomes else "MISSING", "eligibility": "PRESENT" if eligibility.get("eligibilityCriteria") else "MISSING", "enrollment": "PRESENT" if _optional_nonnegative_int(enrollment.get("count")) is not None else "MISSING"}


def _clinicaltrials_version(record: DatasetRecord) -> str:
    return str(record.metadata.get("study_version_time", "") or "")


def _append_clinical_version(selected: DatasetRecord, discarded: DatasetRecord) -> None:
    versions = selected.metadata.setdefault("alternate_versions", [])
    version = _clinicaltrials_version(discarded)
    if version and version not in versions:
        versions.append(version)


DATASET_ADAPTERS = {
    "clinicaltrials": ClinicalTrialsDatasetAdapter,
    "ena": ENASRADatasetAdapter,
    "geo": GEODatasetAdapter,
    "mgnify": MGnifyDatasetAdapter,
}

LITERATURE_ADAPTERS = {
    "crossref": CrossrefLiteratureAdapter,
    "europepmc": EuropePMCLiteratureAdapter,
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
    seen: dict[str, JsonDict] = {}
    for adapter in build_literature_adapters(source_names, client=client):
        for row in adapter.search_literature(query, limit=limit):
            key = _literature_identity(row)
            if key in seen:
                _merge_literature_duplicate(seen[key], row)
                continue
            seen[key] = row
            rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows


def cached_client(cache_dir: str | Path | None = None, *, offline: bool = False) -> CachedHttpClient:
    """Return a cached HTTP client for CLI adapter commands."""

    return CachedHttpClient(cache_dir=cache_dir or Path("local/http_cache"), offline=offline)


def _literature_identity(row: JsonDict) -> str:
    """Use DOI first so the same work from separate sources is merged."""

    doi = _normalize_doi(row.get("doi", ""))
    return f"doi:{doi}" if doi else str(row.get("source_id", "") or row.get("title", "")).strip().lower()


def _client_response_metadata(client: object) -> JsonDict:
    """Return optional cache provenance without constraining test clients."""

    metadata = getattr(client, "last_response_metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _merge_literature_duplicate(existing: JsonDict, duplicate: JsonDict) -> None:
    """Preserve cross-source identifiers and version relations on a DOI merge."""

    metadata = existing.setdefault("metadata", {})
    alternate_ids = metadata.setdefault("alternate_source_ids", [])
    duplicate_id = str(duplicate.get("source_id", "") or "")
    if duplicate_id and duplicate_id != existing.get("source_id") and duplicate_id not in alternate_ids:
        alternate_ids.append(duplicate_id)
    relations = metadata.setdefault("version_relationships", {})
    incoming = duplicate.get("version_relationships") or duplicate.get("metadata", {}).get("version_relationships", {})
    if isinstance(incoming, dict):
        relations.setdefault(str(duplicate.get("source", "") or "unknown"), incoming)
    provenances = metadata.setdefault("alternate_source_provenance", [])
    provenance = duplicate.get("source_provenance", {})
    if provenance and provenance not in provenances:
        provenances.append(provenance)


def _openalex_abstract(index: dict[str, list[int]]) -> str:
    """Reconstruct an OpenAlex inverted-index abstract."""

    if not index:
        return ""
    tokens: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            tokens.append((int(position), word))
    return " ".join(word for _, word in sorted(tokens))


def _normalize_doi(value: object) -> str:
    """Return a canonical DOI token without resolver prefixes."""

    doi = str(value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi


def _first_sequence_text(value: object) -> str:
    """Return the first nonblank string from an API list or scalar."""

    if isinstance(value, (list, tuple)):
        for item in value:
            text = str(item or "").strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _year_from_value(value: object) -> int | None:
    """Extract a plausible four-digit year from a source date field."""

    match = re.search(r"(?<!\d)(\d{4})(?!\d)", str(value or ""))
    return int(match.group(1)) if match else None


def _crossref_year(item: JsonDict) -> int | None:
    """Prefer Crossref publication dates while tolerating incomplete metadata."""

    for key in ("published-online", "published-print", "published", "created"):
        value = item.get(key, {})
        if not isinstance(value, dict):
            continue
        parts = value.get("date-parts", [])
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            try:
                year = int(parts[0][0])
            except (TypeError, ValueError):
                continue
            if 1000 <= year <= 9999:
                return year
    return None


def _crossref_authors(value: object) -> list[str]:
    """Normalize Crossref author dictionaries without manufacturing names."""

    authors: list[str] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        name = " ".join(str(item.get(key, "") or "").strip() for key in ("given", "family")).strip()
        if name:
            authors.append(name)
    return authors


def _strip_jats(value: str) -> str:
    """Reduce Crossref's occasional lightweight JATS markup to text."""

    try:
        return _xml_text(ET.fromstring(value))
    except ET.ParseError:
        return " ".join(value.replace("<", " ").replace(">", " ").split())


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


def _string_list(value: object) -> list[str]:
    """Normalize a source list while preserving nonblank values and order."""

    values = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _first_int(payload: JsonDict, *keys: str) -> int:
    """Return the first parseable integer from possible API field names."""

    for key in keys:
        value = payload.get(key, "")
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0
