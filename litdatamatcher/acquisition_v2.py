"""Bounded, independently replayable real-source acquisition (schema v2).

Source bytes are immutable SHA256 objects outside the source distribution.
Acquisition does not infer biological eligibility or donor identity from titles.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests

VERSION = "acquisition-v2.1"
RESERVED_STUDIES = {"GSE112372": "final_primary_holdout", "GSE214695": "transfer_holdout", "GSE226875": "transfer_holdout"}
LITERATURE_QUERIES = {
    "primary": '(human AND (lipopolysaccharide OR inflammatory) AND (transcriptomic OR "gene expression") AND (stimulation OR treatment)) AND OPEN_ACCESS:Y AND FIRST_PDATE:[2015-01-01 TO 2024-12-31]',
    "transfer": '(human AND ("inflammatory bowel disease" OR "ulcerative colitis") AND (transcriptomic OR "gene expression")) AND OPEN_ACCESS:Y AND FIRST_PDATE:[2015-01-01 TO 2024-12-31]',
}
DATASET_QUERIES = {
    "primary": 'Homo sapiens[Organism] AND (lipopolysaccharide OR inflammatory) AND ("Expression profiling by array"[DataSet Type] OR "Expression profiling by high throughput sequencing"[DataSet Type]) AND gse[Entry Type] AND 2015:2024[Publication Date]',
    "transfer": 'Homo sapiens[Organism] AND ("inflammatory bowel disease" OR "ulcerative colitis") AND ("Expression profiling by array"[DataSet Type] OR "Expression profiling by high throughput sequencing"[DataSet Type]) AND gse[Entry Type] AND 2015:2024[Publication Date]',
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


class SnapshotClient:
    """Request-addressed index into hash-verified immutable responses.

    Offline cache misses and corruption fail before constructing any session.
    Refresh never overwrites an old object. One writer owns each stage/root.
    """
    def __init__(self, root, *, offline=False, refresh=False, session=None, sleep=time.sleep, max_bytes=40_000_000):
        self.root = Path(root)
        self.offline, self.refresh = offline, refresh
        self.session, self.sleep, self.max_bytes = session, sleep, max_bytes
        self.events = []

    def get(self, url, params=None):
        full = url + (("&" if "?" in url else "?") + urlencode(sorted(params.items())) if params else "")
        key = sha256(full.encode())
        index = self.root / "requests" / (key + ".json")
        if index.exists() and (self.offline or not self.refresh):
            meta = json.loads(index.read_text(encoding="utf-8"))
            data = (self.root / "objects" / meta["sha256"]).read_bytes()
            if sha256(data) != meta["sha256"]:
                raise ValueError("corrupt snapshot: " + key)
            self.events.append({"url": full, "status": "offline_replay" if self.offline else "cache_hit", "sha256": meta["sha256"]})
            return data, meta
        if self.offline:
            raise FileNotFoundError("offline snapshot missing: " + full)
        session = self.session or requests.Session()
        last_error = None
        for attempt in range(1, 4):
            try:
                response = session.get(full, timeout=(15, 60), stream=True, headers={"User-Agent": "LitDataMatcher/2 source-research-client"})
                with response:
                    status = response.status_code
                    if status == 429 or 500 <= status < 600:
                        retry = response.headers.get("Retry-After", "")
                        delay = min(10, float(retry)) if re.fullmatch(r"\d+(\.\d+)?", retry) else min(8, 2 ** attempt)
                        self.events.append({"url": full, "status": status, "attempt": attempt, "retry_seconds": delay})
                        response.raise_for_status()
                    response.raise_for_status()
                    chunks, total = [], 0
                    for chunk in response.iter_content(65536):
                        total += len(chunk)
                        if total > self.max_bytes:
                            raise ValueError("bounded download limit exceeded")
                        chunks.append(chunk)
                    data = b"".join(chunks)
                    digest = sha256(data)
                    obj = self.root / "objects" / digest
                    obj.parent.mkdir(parents=True, exist_ok=True)
                    if not obj.exists():
                        with obj.open("xb") as handle:
                            handle.write(data)
                    meta = {"url": full, "sha256": digest, "size_bytes": len(data), "retrieved_at": datetime.now(timezone.utc).isoformat(), "status_code": status, "attempts": attempt, "content_type": response.headers.get("Content-Type"), "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"), "object_path": str(obj.resolve())}
                    write_json(index, meta)
                    self.events.append(meta)
                    return data, meta
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
                last_error, delay = exc, min(8, 2 ** attempt)
            except requests.HTTPError as exc:
                if exc.response is None or (exc.response.status_code != 429 and exc.response.status_code < 500):
                    raise
                last_error = exc
            if attempt < 3:
                self.sleep(delay)
        raise RuntimeError(f"retrieval exhausted after 3 attempts: {full}") from last_error

    def json(self, url, params=None):
        data, meta = self.get(url, params)
        return json.loads(data), meta


def parse_article(data: bytes) -> dict:
    """Nonoverlapping paragraph spans, with captions/tables retained separately."""
    root = ET.fromstring(data)
    if root.tag != "article":
        raise ValueError("expected JATS article")
    body = root.find("body")
    if body is None:
        raise ValueError("JATS body missing")
    blocks, spans, cursor = [], [], 0
    parents = {child: parent for parent in root.iter() for child in parent}
    for index, node in enumerate(body.iter("p")):
        text = " ".join("".join(node.itertext()).split())
        if not text:
            continue
        section = parents.get(node)
        while section is not None and section.tag != "sec":
            section = parents.get(section)
        heading = " ".join("".join(section.find("title").itertext()).split()) if section is not None and section.find("title") is not None else "body"
        spans.append({"start": cursor, "end": cursor + len(text), "text": text, "section": heading, "locator": f"body//p[{index + 1}]"})
        blocks.append(text)
        cursor += len(text) + 1
    text = "\n".join(blocks)
    if len(text) < 200:
        raise ValueError("full text has insufficient parsed body")
    return {"text": text, "text_sha256": sha256(text.encode()), "sections": spans, "license": [" ".join("".join(x.itertext()).split()) for x in root.iter("license")], "article_type": root.get("article-type"), "fulltext_status": "parsed"}


def sync_literature(root, limit=200, fulltexts=50, offline=False, refresh=False):
    root = Path(root)
    client = SnapshotClient(root / "snapshots" / "literature", offline=offline, refresh=refresh)
    records, seen, failures, searches = [], set(), [], []
    quota = {"primary": math.ceil(limit * .7), "transfer": limit - math.ceil(limit * .7)}
    for topic, query in LITERATURE_QUERIES.items():
        cursor, selected = "*", 0
        for _ in range(10):
            payload, meta = client.json("https://www.ebi.ac.uk/europepmc/webservices/rest/search", {"query": query, "format": "json", "resultType": "core", "pageSize": 100, "cursorMark": cursor})
            searches.append({"topic": topic, "query": query, "hit_count": payload["hitCount"], "snapshot": meta})
            for item in payload["resultList"]["result"]:
                identity = "doi:" + item["doi"].lower() if item.get("doi") else item.get("pmcid") or item["source"] + ":" + item["id"]
                if identity in seen:
                    continue
                seen.add(identity)
                records.append({"document_id": identity, "title": item.get("title", ""), "abstract": item.get("abstractText", ""), "text": item.get("abstractText", ""), "doi": item.get("doi"), "pmid": item.get("id") if item.get("source") == "MED" else None, "pmcid": item.get("pmcid"), "publication_date": item.get("firstPublicationDate"), "topic": topic, "split_context": "transfer" if topic == "transfer" else "development", "source": "EuropePMC", "source_locator": "https://europepmc.org/article/" + item["source"] + "/" + item["id"], "source_snapshot": meta, "access_status": "open_access" if item.get("isOpenAccess") == "Y" else "unknown", "fulltext_status": "not_requested", "version_relationships": item.get("commentCorrectionList", {}), "schema_version": VERSION})
                selected += 1
                if selected >= quota[topic]:
                    break
            if selected >= quota[topic] or not payload.get("nextCursorMark") or cursor == payload["nextCursorMark"]:
                break
            cursor = payload["nextCursorMark"]
    # Parse source-selected records in order with a quota per context.
    parsed = {"primary": 0, "transfer": 0}
    text_quotas = {"primary": math.ceil(fulltexts * .7), "transfer": fulltexts - math.ceil(fulltexts * .7)}
    for record in records:
        topic = record["topic"]
        if parsed[topic] >= text_quotas[topic] or not record["pmcid"] or record["access_status"] != "open_access":
            continue
        try:
            data, meta = client.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{record['pmcid']}/fullTextXML")
            record.update(parse_article(data))
            record["fulltext_snapshot"] = meta
            parsed[topic] += 1
        except (ValueError, RuntimeError, requests.RequestException, FileNotFoundError, ET.ParseError) as exc:
            record["fulltext_status"] = "failed"
            failures.append({"id": record["document_id"], "error": str(exc)})
        write_jsonl(root / "catalog" / "literature.jsonl", records)
    write_jsonl(root / "catalog" / "literature.jsonl", records)
    report = {"schema_version": VERSION, "requested_records": limit, "unique_records": len(records), "requested_fulltexts": fulltexts, "parsed_fulltexts": sum(parsed.values()), "parsed_by_topic": parsed, "failures": failures, "searches": searches, "events": client.events, "offline": offline, "status": "PASS" if len(records) >= limit and sum(parsed.values()) >= fulltexts else "FAIL"}
    write_json(root / "catalog" / "literature_acquisition.json", report)
    return records, report


def soft_fields(text: str, prefix: str) -> dict:
    fields = {}
    for line in text.splitlines():
        if line.startswith(prefix) and " = " in line:
            key, value = line.split(" = ", 1)
            fields.setdefault(key[len(prefix):], []).append(value)
    return fields


def parse_series_matrix(data: bytes, accession: str) -> tuple[list, dict]:
    if data[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
            data = handle.read(250_000_001)
        if len(data) > 250_000_000:
            raise ValueError("decompressed matrix exceeds bound")
    rows = csv.reader(io.StringIO(data.decode("utf-8-sig")), delimiter="\t")
    metadata, header, features, numeric, missing, bad, matrix = [], [], [], 0, 0, 0, False
    values_preview = []
    for row in rows:
        if not row:
            continue
        if row[0].startswith("!Sample_"):
            metadata.append((row[0][8:], row[1:]))
        elif row[0] == "!series_matrix_table_begin":
            matrix = True
        elif row[0] == "!series_matrix_table_end":
            matrix = False
        elif matrix and not header:
            header = row[1:]
        elif matrix:
            if len(row) != len(header) + 1:
                raise ValueError("matrix row width mismatch")
            features.append(row[0])
            vals = []
            for cell in row[1:]:
                if cell in ("", "null", "NA", "NaN", "nan"):
                    missing += 1
                    vals.append(None)
                else:
                    try:
                        number = float(cell)
                        if not math.isfinite(number):
                            raise ValueError("non-finite matrix measurement")
                        numeric += 1
                        vals.append(number)
                    except ValueError:
                        bad += 1
                        vals.append(None)
            if len(values_preview) < 5:
                values_preview.append({"feature_id": row[0], "values": vals})
    ids = next((values for key, values in metadata if key == "geo_accession"), [])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicate sample accessions")
    if any(len(values) != len(ids) for _, values in metadata):
        raise ValueError("sample annotation width mismatch")
    samples = []
    for index, sample_id in enumerate(ids):
        fields = {}
        for key, values in metadata:
            fields.setdefault(key, []).append(values[index])
        characteristics = {}
        for value in fields.get("characteristics_ch1", []):
            if ": " in value:
                key, val = value.split(": ", 1)
                characteristics.setdefault(key.lower(), []).append(val)
        donor = next((characteristics[k][0] for k in ("donor", "donor id", "subject id", "patient id") if k in characteristics), None)
        group = next((characteristics[k][0] for k in ("treatment", "condition", "disease state", "disease status", "group", "stimulus") if k in characteristics), None)
        samples.append({"sample_id": sample_id, "study_id": accession, "fields": fields, "characteristics": characteristics, "donor_id": donor, "group": group, "source_locator": f"{accession}:!Sample_*:column={index+2}", "inference_method": "explicit_attribute_keys_only"})
    inspection = {"dataset_id": accession, "sample_count": len(ids), "feature_count": len(features), "unique_features": len(set(features)), "sample_ids": ids, "matrix_sample_ids": header, "feature_space": "submitted_platform_feature_identifiers", "sample_alignment": bool(header) and header == ids, "numeric_cells": numeric, "missing_cells": missing, "invalid_cells": bad, "processed_measurements_present": bool(features) and numeric > 0 and bad == 0, "feature_ids_sha256": sha256("\n".join(features).encode()), "values_preview": values_preview, "normalization": dict(metadata).get("data_processing", []), "units": "source_declared_processing; not assumed comparable across studies", "status": "PASS" if features and numeric and not bad and header == ids and len(set(features)) == len(features) else "METADATA_ONLY_OR_INVALID"}
    return samples, inspection


def _capability(value, locator, reason="not reported in acquired structured metadata"):
    return {"value": value, "status": "known" if value is not None else "unknown", "source_locator": locator, "reason": None if value is not None else reason}


def profile_capabilities(record, samples):
    locator = record["source_locator"]
    capabilities = {key: _capability(None, locator) for key in ("species", "tissue", "assay", "intervention", "comparator", "outcome", "paired", "timepoint")}
    capabilities["species"] = _capability(record.get("organism"), locator)
    capabilities["assay"] = _capability(record.get("assay"), locator)
    for target, keys in {"tissue": ("tissue", "cell type", "cell line"), "intervention": ("treatment", "stimulus", "stimulation"), "timepoint": ("time", "time point", "timepoint")}.items():
        values = sorted({v for s in samples for key in keys for v in s["characteristics"].get(key, []) if v})
        if values:
            capabilities[target] = _capability(values, [s["source_locator"] for s in samples])
    donors = {s["donor_id"] for s in samples if s.get("donor_id")}
    record.update({"samples": samples, "capabilities": capabilities, "independent_units": len(donors) if donors and all(s.get("donor_id") for s in samples) else None, "independent_unit_status": "explicit_sample_donor_ids" if donors and all(s.get("donor_id") for s in samples) else "unknown", "groups": sorted({s["group"] for s in samples if s.get("group")}), "profile_status": "sample_annotations_parsed" if samples else "metadata_only"})


def sync_datasets(root, limit=100, profiles=30, offline=False, refresh=False):
    root = Path(root)
    client = SnapshotClient(root / "snapshots" / "datasets", offline=offline, refresh=refresh)
    records, seen, searches, failures, inspections, ena_rows = [], set(), [], [], [], []
    for topic, query in DATASET_QUERIES.items():
        payload, search_meta = client.json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", {"db": "gds", "term": query, "retmode": "json", "retmax": math.ceil(limit * (.8 if topic == "primary" else .5))})
        ids = payload["esearchresult"]["idlist"]
        searches.append({"topic": topic, "query": query, "hit_count": int(payload["esearchresult"]["count"]), "snapshot": search_meta})
        for start in range(0, len(ids), 20):
            if not offline:
                time.sleep(.4)
            result, meta = client.json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", {"db": "gds", "id": ",".join(ids[start:start+20]), "retmode": "json"})
            for uid in result["result"]["uids"]:
                item = result["result"][uid]
                accession = item.get("accession", "")
                if not accession.startswith("GSE") or accession in seen:
                    continue
                seen.add(accession)
                record = {"schema_version": VERSION, "dataset_id": accession, "title": item.get("title", ""), "summary": item.get("summary", ""), "organism": item.get("taxon"), "assay": item.get("gdstype"), "topic": topic, "split_context": "transfer" if topic == "transfer" else "development", "source": "GEO", "source_locator": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=" + accession, "source_snapshot": meta, "study_lineage": [accession], "repository_aliases": [], "sample_count_reported": item.get("n_samples"), "discovery_samples": item.get("samples", []), "samples": [], "access_status": "public", "processed_availability": "uninspected", "publication_ids": item.get("pubmedids", []), "independence": "distinct_accession; cohort_overlap_unresolved"}
                profile_capabilities(record, [])
                records.append(record)
    write_jsonl(root / "catalog" / "studies.jsonl", records)
    # The sample budget is source-order based, capped at 250 samples per study.
    # Metadata-only matrices are valid profiles but not processed-file successes.
    profile_quotas = {"primary": math.ceil(profiles * .7), "transfer": profiles - math.ceil(profiles * .7)}
    for record in records:
        topic_profiled = sum(bool(r["samples"]) and r["topic"] == record["topic"] for r in records)
        accession = record["dataset_id"]
        try:
            data, meta = client.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", {"acc": accession, "targ": "self", "form": "text", "view": "full"})
            fields = soft_fields(data.decode("utf-8"), "!Series_")
            record["series_metadata"] = fields
            record["series_snapshot"] = meta
            aliases = sorted(set(re.findall(r"\b(?:PRJNA\d+|SRP\d+|ERP\d+|PRJEB\d+)\b", " ".join(fields.get("relation", [])))))
            record["repository_aliases"] = aliases
            related = sorted(set(re.findall(r"\bGSE\d+\b", " ".join(fields.get("relation", [])))))
            record["related_series"] = related
            record["study_lineage"] = sorted(set([accession] + aliases + related))
            record["publication_ids"] = fields.get("pubmed_id", [])
            if aliases and not ena_rows:
                try:
                    rows, ena_meta = client.json("https://www.ebi.ac.uk/ena/portal/api/filereport", {"accession": aliases[0], "result": "read_run", "fields": "study_accession,secondary_study_accession,sample_accession,run_accession,scientific_name,library_strategy", "format": "json", "limit": 1000})
                    ena_rows.extend({**row, "canonical_dataset_id": accession, "counts_as_additional_study": False, "source_snapshot": ena_meta} for row in rows)
                except (ValueError, RuntimeError, requests.RequestException, FileNotFoundError) as exc:
                    failures.append({"id": accession, "stage": "ena", "error": str(exc)})
            if topic_profiled >= profile_quotas[record["topic"]] and len({x["dataset_id"] for x in inspections if x["status"] == "PASS"}) >= 2 and ena_rows:
                continue
            if len(record["discovery_samples"]) > 250:
                failures.append({"id": accession, "stage": "profile", "status": "bounded_skip", "reason": "more than 250 discovery samples"})
                continue
            prefix = accession[:-3] + "nnn" if len(accession) > 6 else "GSEnnn"
            url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{accession}/matrix/"
            listing, _ = client.get(url)
            files = sorted(set(re.findall(r'href="([^"/]+series_matrix[^"/]*\.txt\.gz)"', listing.decode())))
            if not files:
                raise ValueError("no series matrix in directory")
            for filename in files[:3]:
                matrix, matrix_meta = client.get(url + filename)
                samples, inspection = parse_series_matrix(matrix, accession)
                inspection["source_snapshot"] = matrix_meta
                inspection["source_url"] = url + filename
                inspections.append(inspection)
                profile_capabilities(record, record["samples"] + [s for s in samples if s["sample_id"] not in {x["sample_id"] for x in record["samples"]}])
                record["processed_availability"] = "inspected" if inspection["status"] == "PASS" else "metadata_matrix_only"
                record.setdefault("processed_inspections", []).append(inspection)
        except (ValueError, RuntimeError, requests.RequestException, FileNotFoundError, OSError) as exc:
            failures.append({"id": accession, "stage": "profile", "error": str(exc)})
        write_jsonl(root / "catalog" / "studies.jsonl", records)
        write_jsonl(root / "catalog" / "processed_inspections.jsonl", inspections)
        write_jsonl(root / "catalog" / "ena_runs.jsonl", ena_rows)
    # Explicit BioProject overlaps count once; lack of a cross-reference does not prove independence.
    parents = {r["dataset_id"]: r["dataset_id"] for r in records}
    def find(accession):
        while parents[accession] != accession:
            accession = parents[accession]
        return accession
    aliases_seen = {}
    for r in records:
        # A common explicitly related series, BioProject or sample marks dependence.
        identifiers = r["study_lineage"] + [x["accession"] for x in r["discovery_samples"]]
        for alias in identifiers:
            if alias in aliases_seen:
                parents[find(r["dataset_id"])] = find(aliases_seen[alias])
            else:
                aliases_seen[alias] = r["dataset_id"]
    for r in records:
        r["dependence_group"] = find(r["dataset_id"])
        linked_reserves = [value for key, value in RESERVED_STUDIES.items() if key in r["study_lineage"]]
        r["reserved_evaluation"] = linked_reserves or None
    write_jsonl(root / "catalog" / "studies.jsonl", records)
    unique = len({find(x) for x in parents})
    profiled = sum(bool(r["samples"]) for r in records)
    processed = len({x["dataset_id"] for x in inspections if x["status"] == "PASS"})
    report = {"schema_version": VERSION, "requested_studies": limit, "accession_records": len(records), "unique_study_groups": unique, "cohort_overlap": "unresolved except explicit shared BioProject aliases; no claim all independent cohorts", "requested_profiles": profiles, "sample_profiles": profiled, "processed_studies": processed, "ena_run_metadata_records": len(ena_rows), "ena_additional_studies_counted": 0, "searches": searches, "failures": failures, "events": client.events, "offline": offline, "status": "PASS" if unique >= limit and profiled >= profiles and processed >= 2 and ena_rows else "FAIL"}
    write_json(root / "catalog" / "dataset_acquisition.json", report)
    return records, report
