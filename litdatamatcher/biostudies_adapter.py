"""Bounded BioStudies/ArrayExpress metadata discovery through the public API."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from urllib.parse import urlencode

from .datasets import classify_dataset_record
from .http_cache import CachedHttpClient
from .omics_adapters import _provenance


@dataclass(slots=True)
class BioStudiesDatasetAdapter:
    client: CachedHttpClient
    name: str = "biostudies"
    collection: str | None = None
    last_search_status: dict = field(default_factory=dict, init=False)

    def search(self, query: str):
        endpoint = "https://www.ebi.ac.uk/biostudies/api/v1/search"
        params = {"query": query, "pageSize": 25, "page": 1}
        if self.collection:
            params["collection"] = self.collection
        data = self.client.get_json(endpoint, params=params)
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("hits"), list)
            or type(data.get("totalHits")) is not int
            or data["totalHits"] < 0
        ):
            raise ValueError("BioStudies schema drift: expected hits and totalHits")
        records, seen = [], set()
        for index, item in enumerate(data["hits"]):
            if not isinstance(item, dict):
                raise ValueError("BioStudies hit must be an object")
            accession = item.get("accession", "")
            if (
                not isinstance(accession, str)
                or not re.fullmatch(r"[A-Z][A-Z0-9-]{2,80}", accession)
                or not isinstance(item.get("title"), str)
                or not item["title"].strip()
                or accession in seen
            ):
                raise ValueError("Invalid or duplicate BioStudies study identity")
            seen.add(accession)
            url = f"https://www.ebi.ac.uk/biostudies/studies/{accession}"
            provenance = _provenance(
                self.client,
                self.name,
                accession,
                url,
                "biostudies-search-v1",
                [
                    "Search hits may be literature-associated studies without reusable measurement files.",
                    "File counts do not establish processed data availability, licensing, or scientific suitability.",
                ],
            )
            provenance["source_locator"] = f"{endpoint}?{urlencode(params)}#/hits/{index}"
            records.append(
                classify_dataset_record(
                    {
                        "dataset_id": accession,
                        "source": "BioStudies",
                        "title": html.unescape(re.sub(r"<[^>]*>", "", item["title"])),
                        "description": html.unescape(
                            re.sub(r"<[^>]*>", "", str(item.get("content", "")))
                        ),
                        "url": url,
                        "access_type": "public metadata"
                        if item.get("isPublic") is True
                        else "unknown",
                        "metadata": {
                            "source_provenance": provenance,
                            "source_hit": item,
                            "reported_file_count": item.get("files"),
                            "collection": self.collection,
                            "measurement_availability": "UNKNOWN_UNINSPECTED",
                        },
                    }
                )
            )
        self.last_search_status = {
            "status": "BOUNDED_RESULTS",
            "returned_items": len(records),
            "reported_total_hits": data["totalHits"],
            "page": data.get("page", 1),
            "candidate_universe_status": "PARTIAL_CANDIDATE_UNIVERSE_NOT_EVIDENCE_COMPLETE",
        }
        return records


@dataclass(slots=True)
class ArrayExpressDatasetAdapter(BioStudiesDatasetAdapter):
    name: str = "arrayexpress"
    collection: str | None = "arrayexpress"
