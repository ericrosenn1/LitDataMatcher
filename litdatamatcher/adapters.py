"""Optional live literature and dataset adapters.

The default pipeline remains offline and deterministic. These adapters provide
the scaffolding needed to extend LitDataMatcher toward live repository scraping
while preserving caching, provenance, and schema normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .datasets import classify_dataset_record
from .http_cache import CachedHttpClient
from .schemas import DatasetRecord


class LiteratureSourceAdapter(Protocol):
    """Protocol for literature search adapters."""

    name: str

    def search_literature(self, query: str, limit: int = 25) -> list[dict]:
        """Return literature records with title, abstract, doi, and source_id."""


@dataclass(slots=True)
class OpenAlexLiteratureAdapter:
    """OpenAlex literature metadata adapter."""

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
            rows.append(
                {
                    "source_id": work.get("id", ""),
                    "title": work.get("title") or work.get("display_name") or "",
                    "abstract": abstract,
                    "doi": (work.get("doi") or "").replace("https://doi.org/", ""),
                    "year": work.get("publication_year"),
                    "source": self.name,
                    "metadata": {
                        "cited_by_count": work.get("cited_by_count", 0),
                        "concepts": [
                            concept.get("display_name", "")
                            for concept in work.get("concepts", [])[:10]
                        ],
                    },
                }
            )
        return rows


@dataclass(slots=True)
class ClinicalTrialsDatasetAdapter:
    """ClinicalTrials.gov study metadata adapter."""

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
            records.append(
                classify_dataset_record(
                    {
                        "dataset_id": identification.get("nctId", ""),
                        "title": title,
                        "source": "ClinicalTrials.gov",
                        "description": description.get("briefSummary", ""),
                        "url": f"https://clinicaltrials.gov/study/{identification.get('nctId', '')}",
                        "variables": variables,
                        "populations": ["human"],
                        "assay_types": ["clinical registry"],
                        "sample_size": design.get("enrollmentInfo", {}).get("count", 0),
                        "license": "public domain US government work",
                        "access_type": "public",
                        "quality_score": 0.72,
                        "metadata": study,
                    }
                )
            )
        return records


def _openalex_abstract(index: dict[str, list[int]]) -> str:
    """Reconstruct an OpenAlex inverted-index abstract."""

    if not index:
        return ""
    tokens: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            tokens.append((int(position), word))
    return " ".join(word for _, word in sorted(tokens))
