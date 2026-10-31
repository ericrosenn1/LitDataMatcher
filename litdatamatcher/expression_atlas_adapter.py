"""Search the cached public Expression Atlas experiment catalogue."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .datasets import classify_dataset_record
from .http_cache import CachedHttpClient
from .omics_adapters import _provenance


@dataclass(slots=True)
class ExpressionAtlasDatasetAdapter:
    client: CachedHttpClient
    name: str = "expressionatlas"
    last_search_status: dict = field(default_factory=dict, init=False)

    def search(self, query: str):
        endpoint = "https://www.ebi.ac.uk/gxa/json/experiments"
        data = self.client.get_json(endpoint)
        if not isinstance(data, dict) or not isinstance(data.get("experiments"), list):
            raise ValueError("Expression Atlas catalogue schema drift")
        experiments = data["experiments"]
        if len(experiments) > 50_000:
            raise ValueError("Expression Atlas catalogue exceeds inspection bound")
        terms = re.findall(r"[a-z0-9]+", query.casefold())
        if not terms:
            raise ValueError("A nonempty query is required")
        records, matched, seen = [], 0, set()
        for index, item in enumerate(experiments):
            if not isinstance(item, dict):
                raise ValueError("Expression Atlas experiment must be an object")
            accession = item.get("experimentAccession", "")
            if (
                not isinstance(accession, str)
                or not re.fullmatch(r"E-[A-Z]+-\d+", accession)
                or accession in seen
                or not isinstance(item.get("experimentDescription"), str)
            ):
                raise ValueError("Expression Atlas experiment identity/schema drift")
            seen.add(accession)
            text = " ".join(
                str(item.get(key, ""))
                for key in (
                    "experimentAccession",
                    "experimentDescription",
                    "species",
                    "experimentalFactors",
                )
            )
            tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
            if not all(term in tokens for term in terms):
                continue
            matched += 1
            if len(records) >= 25:
                continue
            url = "https://www.ebi.ac.uk/gxa/experiments/" + accession
            locator = f"{endpoint}#/experiments/{index}"
            provenance = _provenance(
                self.client,
                self.name,
                accession,
                url,
                "expression-atlas-catalogue-v1",
                [
                    "Search is a bounded lexical filter over the cached experiment catalogue, not a gene-expression query.",
                    "Assay counts are technical units; measurements and biological independence remain uninspected.",
                ],
            )
            provenance["source_locator"] = locator
            species = item.get("species")
            organisms = [species] if isinstance(species, str) and species.strip() else []
            assays = item.get("technologyType", [])
            if not isinstance(assays, list) or any(not isinstance(value, str) for value in assays):
                raise ValueError("Expression Atlas technology type schema drift")
            metadata = {
                "source_provenance": provenance,
                "source_record": item,
                "reported_assay_count": item.get("numberOfAssays"),
                "experimental_factors": item.get("experimentalFactors", []),
                "measurement_availability": "UNKNOWN_UNINSPECTED",
                "field_evidence": {},
            }
            if organisms:
                metadata["field_evidence"]["organisms"] = {
                    "value": organisms,
                    "status": "observed",
                    "source_locator": locator + "/species",
                    "mapping_type": "exact",
                }
            if assays:
                metadata["field_evidence"]["assay_types"] = {
                    "value": assays,
                    "status": "observed",
                    "source_locator": locator + "/technologyType",
                    "mapping_type": "exact",
                }
            records.append(
                classify_dataset_record(
                    {
                        "dataset_id": accession,
                        "source": "Expression Atlas",
                        "title": item["experimentDescription"],
                        "description": item["experimentDescription"],
                        "url": url,
                        "organisms": organisms,
                        "assay_types": assays,
                        "access_type": "public metadata; processed files require inspection",
                        "metadata": metadata,
                    }
                )
            )
        self.last_search_status = {
            "status": "BOUNDED_RESULTS",
            "returned_items": len(records),
            "catalogue_records": len(experiments),
            "matching_catalogue_records": matched,
            "search_method": "all-query-tokens-in-catalogue-metadata",
            "candidate_universe_status": "PARTIAL_CANDIDATE_UNIVERSE_NOT_EVIDENCE_COMPLETE",
        }
        return records
