"""Bounded public proteomics/metabolomics metadata routes; no measurements inferred."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote

from .datasets import classify_dataset_record
from .http_cache import CachedHttpClient
from .provenance import remote_source_provenance


def _strings(value):
    if isinstance(value, str):
        value = [value]
    return sorted({item.strip() for item in value or [] if isinstance(item, str) and item.strip()})


def _provenance(client, name, identity, url, version, limitations):
    snapshot = dict(getattr(client, "last_response_metadata", {}))
    return remote_source_provenance(
        source_type=name, source_url=url, adapter_name=name, adapter_version=version,
        retrieval_time_utc=snapshot.get("retrieval_time_utc", ""),
        acquisition_method="public_metadata_api", content_scope="study_metadata_only",
        raw_record_id=identity, limitations=limitations,
        next_handoff="inspect experimental compatibility and source files",
        metadata={"cache_snapshot": snapshot},
    ).to_dict()


@dataclass(slots=True)
class PRIDEDatasetAdapter:
    client: CachedHttpClient
    name: str = "pride"

    def search(self, query: str):
        data = self.client.get_json(
            "https://www.ebi.ac.uk/pride/ws/archive/v2/search/projects",
            params={"keyword": query, "pageSize": 25, "page": 0},
        )
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("PRIDE schema drift: expected project list")
        records = []
        for item in data:
            identity = item.get("accession", "")
            if not isinstance(identity, str) or not re.fullmatch(r"PXD\d{6,}", identity) or not item.get("title"):
                raise ValueError("PRIDE project lacks valid identity/title")
            url = f"https://www.ebi.ac.uk/pride/archive/projects/{identity}"
            provenance = _provenance(self.client, self.name, identity, url, "pride_archive_v2_metadata_1", [
                "Project/sample counts do not establish donors or independent units.",
                "File names are availability metadata, not inspected processed measurements.",
                "Projects may share subjects; independence requires an explicit lineage review.",
            ])
            metadata = {
                "source_provenance": provenance,
                "version_time": item.get("updatedDate", "UNKNOWN"),
                "publication_date": item.get("publicationDate", "UNKNOWN"),
                "tissues": _strings(item.get("organismsPart")),
                "experiment_types": _strings(item.get("experimentTypes")),
                "instruments": _strings(item.get("instruments")),
                "file_names": _strings(item.get("projectFileNames")),
                "publication_references": item.get("references", []),
                "source_protocols": {key: item.get(key, "") for key in ("sampleProcessingProtocol", "dataProcessingProtocol")},
                "omics_contract": {"feature_type": "UNKNOWN", "feature_unit": "UNKNOWN", "quantification": "UNKNOWN", "normalization": "UNKNOWN"},
                "dependence": {"donor_links": "AMBIGUOUS_NOT_INFERRED"},
                "pagination": {"status": "BOUNDED_PAGE_NOT_COMPLETE_CENSUS", "page": 0, "page_size": 25, "returned_items": len(data)},
                "access_status": "PUBLIC_METADATA_ONLY",
                "missingness": {key: "UNKNOWN" for key in ("donors", "comparator", "outcomes", "temporal_design", "processed_alignment")},
            }
            records.append(classify_dataset_record({
                "dataset_id": identity, "title": item["title"], "description": item.get("projectDescription", ""),
                "source": "PRIDE", "url": url, "organisms": _strings(item.get("organisms")),
                "assay_types": ["proteomics"], "sample_size": 0, "variables": [],
                "access_type": "public metadata; file-level inspection required",
                "license": "Source-specific terms require review before redistribution", "metadata": metadata,
            }))
        return records


@dataclass(slots=True)
class MetabolomicsWorkbenchDatasetAdapter:
    client: CachedHttpClient
    name: str = "metabolomicsworkbench"

    def search(self, query: str):
        query = str(query).strip()
        if not query or len(query) > 200:
            raise ValueError("A bounded study ID or title query is required")
        field = "study_id" if re.fullmatch(r"ST\d{6}", query) else "study_title"
        data = self.client.get_json(f"https://www.metabolomicsworkbench.org/rest/study/{field}/{quote(query, safe='')}/summary")
        if isinstance(data, dict) and "study_id" in data:
            items = [data]
        elif isinstance(data, dict) and all(isinstance(value, dict) for value in data.values()):
            items = list(data.values())
        elif isinstance(data, list) and all(isinstance(value, dict) for value in data):
            items = data
        else:
            raise ValueError("Metabolomics Workbench schema drift: expected study summaries")
        records = []
        for item in items[:100]:
            identity = item.get("study_id", "")
            if not isinstance(identity, str) or not re.fullmatch(r"ST\d{6}", identity) or not item.get("study_title"):
                raise ValueError("Metabolomics study lacks valid identity/title")
            url = f"https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?StudyID={identity}"
            provenance = _provenance(self.client, self.name, identity, url, "mw_rest_study_summary_1", [
                "Summary records do not establish independent donors, aligned measurements, or controls.",
                "No sample measurements, raw files, or matrix files were downloaded.",
            ])
            metadata = {
                "source_provenance": provenance, "version_time": item.get("revision_datetime", "UNKNOWN"),
                "source_version": item.get("version", "UNKNOWN"), "source_revision": item.get("revision_no", "UNKNOWN"),
                "publication_date": item.get("release_date", "UNKNOWN"), "analysis_type": item.get("analysis_type", "UNKNOWN"),
                "reported_sample_count": item.get("number_of_samples", "UNKNOWN"),
                "license_url": item.get("license_url", "UNKNOWN"),
                "omics_contract": {"feature_type": "UNKNOWN", "feature_unit": "UNKNOWN", "quantification": "UNKNOWN", "normalization": "UNKNOWN"},
                "dependence": {"donor_links": "AMBIGUOUS_NOT_INFERRED"},
                "pagination": {"status": "BOUNDED_RESPONSE_NOT_COMPLETE_CENSUS", "returned_items": len(items), "retained_limit": 100},
                "access_status": "PUBLIC_METADATA_ONLY",
                "missingness": {key: "UNKNOWN" for key in ("donors", "comparator", "outcomes", "temporal_design", "processed_alignment")},
            }
            records.append(classify_dataset_record({
                "dataset_id": identity, "title": item["study_title"], "description": item.get("study_summary", ""),
                "source": "Metabolomics Workbench", "url": url,
                "organisms": _strings(item.get("species") or item.get("subject_species")),
                "assay_types": ["metabolomics"], "sample_size": 0, "variables": [],
                "access_type": "public metadata; file-level inspection required",
                "license": item.get("license", "UNKNOWN; source-specific terms require review"), "metadata": metadata,
            }))
        return records
