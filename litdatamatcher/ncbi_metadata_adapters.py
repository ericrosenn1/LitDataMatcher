"""BioProject and BioSample discovery without turning metadata into measurements."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from urllib.parse import urlencode

from .datasets import classify_dataset_record
from .http_cache import CachedHttpClient
from .omics_adapters import _provenance

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def _object(data):
    if not isinstance(data, dict) or any(data.get(key) for key in ("error", "ERROR", "errorlist")):
        raise ValueError("NCBI error response or malformed object")
    return data


def _biosample_attributes(text, accession):
    if not text:
        return {}, []
    if not isinstance(text, str) or "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("Unsafe or malformed BioSample XML")
    root = ET.fromstring(text)
    if root.tag != "BioSample" or root.get("accession") != accession:
        raise ValueError("BioSample XML accession mismatch")
    attributes = {}
    for item in root.findall("./Attributes/Attribute"):
        key = item.get("harmonized_name") or item.get("attribute_name")
        if key and item.text:
            attributes.setdefault(key, []).append(item.text)
    projects = sorted(
        {
            item.get("label")
            for item in root.findall("./Links/Link")
            if item.get("target") == "bioproject"
            and re.fullmatch(r"PRJ[A-Z]{2}\d+", item.get("label", ""))
        }
    )
    return attributes, projects


@dataclass(slots=True)
class BioProjectDatasetAdapter:
    client: CachedHttpClient
    name: str = "bioproject"
    database: str = "bioproject"
    last_search_status: dict = field(default_factory=dict, init=False)

    def search(self, query: str):
        # The existing shared HTTP cache provides retries and immutable response
        # provenance. Retained pages remain bounded even if a query is very broad.
        if hasattr(self.client, "min_interval_seconds"):
            self.client.min_interval_seconds = max(0.35, self.client.min_interval_seconds)
        params = {"db": self.database, "term": query, "retmode": "json", "retmax": 25}
        data = _object(self.client.get_json(BASE + "esearch.fcgi", params=params))
        result = _object(data.get("esearchresult"))
        identifiers = result.get("idlist")
        if (
            not isinstance(identifiers, list)
            or len(identifiers) > 25
            or any(
                not isinstance(item, str) or not re.fullmatch(r"\d+", item) for item in identifiers
            )
            or len(set(identifiers)) != len(identifiers)
            or not re.fullmatch(r"\d+", str(result.get("count", "")))
        ):
            raise ValueError("NCBI search identity/count schema drift")
        self.last_search_status = {
            "status": "BOUNDED_RESULTS",
            "reported_total_hits": int(result["count"]),
            "returned_items": 0,
            "candidate_universe_status": "PARTIAL_CANDIDATE_UNIVERSE_NOT_EVIDENCE_COMPLETE",
        }
        if (int(result["count"]) > 0 and not identifiers) or len(identifiers) > int(
            result["count"]
        ):
            raise ValueError("NCBI search count and returned first page disagree")
        if not identifiers:
            return []
        params = {"db": self.database, "id": ",".join(identifiers), "retmode": "json"}
        data = _object(self.client.get_json(BASE + "esummary.fcgi", params=params))
        result = _object(data.get("result"))
        records, missing, seen = [], [], set()
        for identity in identifiers:
            raw = result.get(identity)
            if not isinstance(raw, dict) or raw.get("error"):
                missing.append(identity)
                continue
            if str(raw.get("uid")) != identity:
                raise ValueError("NCBI summary UID mismatch")
            sample = self.database == "biosample"
            accession = raw.get("accession" if sample else "project_acc", "")
            pattern = r"SAM[A-Z]+\d+" if sample else r"PRJ[A-Z]{2}\d+"
            if (
                not isinstance(accession, str)
                or not re.fullmatch(pattern, accession)
                or accession in seen
            ):
                raise ValueError("Invalid or duplicate NCBI accession")
            seen.add(accession)
            url = f"https://www.ncbi.nlm.nih.gov/{self.database}/{accession}"
            locator = BASE + "esummary.fcgi?" + urlencode(params) + f"#/result/{identity}"
            organism = raw.get("organism" if sample else "organism_name")
            organisms = [organism] if isinstance(organism, str) and organism.strip() else []
            provenance = _provenance(
                self.client,
                self.name,
                accession,
                url,
                "ncbi-linked-metadata-v1",
                [
                    "BioProject records are study metadata; BioSample records describe individual samples.",
                    "Repository metadata do not establish processed measurement access, donor independence, or analysis feasibility.",
                ],
            )
            provenance["source_locator"] = locator
            attributes, projects = (
                _biosample_attributes(raw.get("sampledata"), accession) if sample else ({}, [])
            )
            metadata = {
                "source_provenance": provenance,
                "source_record": raw,
                "entity_type": "sample_metadata" if sample else "project_metadata",
                "reported_attributes": attributes,
                "linked_bioprojects": projects,
                "measurement_availability": "UNKNOWN_UNINSPECTED",
                "field_evidence": {},
            }
            if organisms:
                metadata["field_evidence"]["organisms"] = {
                    "value": organisms,
                    "status": "observed",
                    "source_locator": locator + ("/organism" if sample else "/organism_name"),
                    "mapping_type": "exact",
                }
            records.append(
                classify_dataset_record(
                    {
                        "dataset_id": accession,
                        "source": "BioSample" if sample else "BioProject",
                        "title": raw.get("title" if sample else "project_title") or accession,
                        "description": raw.get("project_description", "")
                        if not sample
                        else raw.get("title", ""),
                        "url": url,
                        "organisms": organisms,
                        "metadata": metadata,
                        "access_type": "public metadata; measurement access unverified",
                    }
                )
            )
        self.last_search_status.update(
            returned_items=len(records),
            missing_ids=missing,
            status="SCHEMA_DRIFT" if missing else "BOUNDED_RESULTS",
        )
        return records


@dataclass(slots=True)
class BioSampleDatasetAdapter(BioProjectDatasetAdapter):
    name: str = "biosample"
    database: str = "biosample"
