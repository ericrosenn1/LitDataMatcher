"""Small versioned UniProt reference import, always contextual to questions.

UniProt Consortium data: CC BY 4.0, https://www.uniprot.org/help/license.
No inference of experimental design or independent support from curated text.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ACCESSIONS = ("P01375", "P05231", "P01584")  # human TNF, IL6, IL1B
BASE_URL = "https://rest.uniprot.org"
LICENSE_URL = BASE_URL + "/help/license"
MAX_BYTES = 2_000_000


class ResourceUnavailable(RuntimeError):
    """Missing, unavailable, corrupt, or incompatible reference prerequisite."""


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".reference-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _fetch(url):
    for attempt in range(3):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "LitDataMatcher/2 reference-import",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                content = response.read(MAX_BYTES + 1)
                if len(content) > MAX_BYTES:
                    raise ResourceUnavailable("Reference response exceeds bounded limit")
                return content, dict(
                    release=response.headers.get("X-UniProt-Release"),
                    release_date=response.headers.get("X-UniProt-Release-Date"),
                )
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise ResourceUnavailable(f"Reference HTTP {exc.code}: {url}") from exc
            retry = exc.headers.get("Retry-After", str(2**attempt))
            try:
                delay = float(retry)
            except ValueError:
                raise ResourceUnavailable("Retry-After requires deferred retry") from exc
            if not 0 <= delay <= 10:
                raise ResourceUnavailable(
                    "Retry-After exceeds bounded foreground retry; defer"
                ) from exc
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 2:
                raise ResourceUnavailable(f"Reference retrieval unavailable: {url}") from exc
            time.sleep(2**attempt)


def _snapshot(root, name, url, offline, refresh, validator=None):
    manifest_path = root / "manifests" / (name + ".json")
    if manifest_path.exists() and not refresh:
        try:
            manifest = json.loads(manifest_path.read_bytes())
            sha = manifest["sha256"]
            if not re.fullmatch(r"[a-f0-9]{64}", sha) or manifest["url"] != url:
                raise ValueError("Invalid reference manifest identity")
            content = (root / "snapshots" / (sha + ".json")).read_bytes()
            if hashlib.sha256(content).hexdigest() != sha:
                raise ValueError("Reference snapshot hash mismatch")
            payload = json.loads(content)
            if validator:
                validator(payload, manifest)
            return payload, manifest
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ResourceUnavailable(
                "Corrupt reference cache; preserve it and explicitly refresh connected"
            ) from exc
    if offline:
        raise ResourceUnavailable(
            "Offline reference cache missing; perform a connected import first"
        )
    content, headers = _fetch(url)
    try:
        payload = json.loads(content)
    except ValueError as exc:
        raise ResourceUnavailable("Reference returned malformed JSON") from exc
    sha = hashlib.sha256(content).hexdigest()
    manifest = dict(
        url=url,
        sha256=sha,
        size_bytes=len(content),
        **headers,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    if validator:
        validator(payload, manifest)
    _atomic(root / "snapshots" / (sha + ".json"), content)
    # Invalid remote schemas never replace a previously validated cache pointer.
    _atomic(manifest_path, _canonical(manifest))
    return payload, manifest


def normalize_entry(payload, manifest):
    if not isinstance(payload, dict):
        raise ResourceUnavailable("Reference entry is not an object")
    accession = payload.get("primaryAccession")
    audit = payload.get("entryAudit", {})
    if (
        not isinstance(accession, str)
        or not re.fullmatch(r"[A-Z0-9]{6,10}", accession)
        or type(audit.get("entryVersion")) is not int
        or not audit.get("lastAnnotationUpdateDate")
        or not manifest.get("release")
    ):
        raise ResourceUnavailable("Reference schema drift: absent identity/version/release")
    organism = payload.get("organism", {})
    if organism.get("taxonId") != 9606:
        raise ResourceUnavailable("Configured reference panel requires human entries")
    names = []
    for gene in payload.get("genes", []):
        for item in [gene.get("geneName", {})] + gene.get("synonyms", []):
            if isinstance(item.get("value"), str):
                names.append(item["value"])
    if not names:
        raise ResourceUnavailable("Reference lacks gene identity")
    annotations = []
    for i, comment in enumerate(payload.get("comments", [])):
        if comment.get("commentType") != "FUNCTION":
            continue
        for j, text in enumerate(comment.get("texts", [])):
            if not isinstance(text.get("value"), str) or not text["value"].strip():
                continue
            evidences = text.get("evidences", [])
            annotations.append(
                dict(
                    text=text["value"], evidence_codes=evidences, locator=f"/comments/{i}/texts/{j}"
                )
            )
    if not annotations:
        raise ResourceUnavailable("Reference has no usable function annotations")
    return dict(
        schema_version="2.0",
        resource="UniProtKB",
        accession=accession,
        entry_type=payload.get("entryType", "UNKNOWN"),
        aliases=sorted(set(names + [accession])),
        organism=organism["scientificName"],
        taxon_id=9606,
        entry_version=audit["entryVersion"],
        last_annotation_update=audit["lastAnnotationUpdateDate"],
        resource_release=manifest["release"],
        annotations=annotations,
        source_manifest=manifest,
        license="CC-BY-4.0",
        attribution="UniProt Consortium; normalized reference annotations, unchanged source text",
        license_url="https://www.uniprot.org/help/license",
        experimental_lineage="UNKNOWN",
    )


def import_resource(data_root, offline=False, *, accessions=DEFAULT_ACCESSIONS, refresh=False):
    """Import/replay the bounded human cytokine panel; failure never means absence."""
    if offline and refresh:
        raise ValueError("Offline refresh would require networking")
    accessions = tuple(sorted(set(accessions)))
    if (
        not accessions
        or len(accessions) > 20
        or any(not isinstance(x, str) or not re.fullmatch(r"[A-Z0-9]{6,10}", x) for x in accessions)
    ):
        raise ValueError("One to twenty safe UniProt accessions required")
    root = Path(data_root).resolve() / "external_evidence" / "uniprot"

    def validate_license(payload, manifest):
        if not isinstance(payload, dict) or "creativecommons.org/licenses/by/4.0" not in str(
            payload.get("content", "")
        ):
            raise ResourceUnavailable("Current/cached reference license not qualified as CC BY 4.0")

    license_payload, license_manifest = _snapshot(
        root, "license", LICENSE_URL, offline, refresh, validate_license
    )
    records = []
    for accession in accessions:

        def validate_entry(payload, manifest, accession=accession):
            if not isinstance(payload, dict) or payload.get("primaryAccession") != accession:
                raise ResourceUnavailable("Reference accession mismatch")
            normalize_entry(payload, manifest)

        payload, manifest = _snapshot(
            root,
            accession,
            BASE_URL + "/uniprotkb/" + accession + ".json",
            offline,
            refresh,
            validate_entry,
        )
        record = normalize_entry(payload, manifest)
        record["license_manifest"] = license_manifest
        records.append(record)
    return records


def query_resource(records, entities, *, proposition_id=None):
    """Exact alias/identifier lookup, yielding context-only compiler records.

    `entities` is a list of gene symbols/accessions, not arbitrary query prose.
    The caller supplies its question proposition ID to link contextual evidence.
    """
    if isinstance(entities, str):
        raise ValueError("entities must be a list, not query prose")

    def key(value):
        return value.casefold().replace("-", "").strip()

    sought = {key(value) for value in entities if isinstance(value, str) and value.strip()}
    results = {}
    for record in records:
        matched = sorted(sought & {key(x) for x in record["aliases"]})
        if not matched:
            continue
        for annotation in record["annotations"]:
            references = annotation["evidence_codes"]
            primary = sorted(
                {
                    "PMID:" + str(x["id"])
                    for x in references
                    if x.get("source") == "PubMed" and x.get("id")
                }
            )
            sid = "UniProtKB:" + record["accession"]
            eid = (
                "reference_"
                + hashlib.sha256(
                    _canonical([sid, record["entry_version"], annotation["locator"]])
                ).hexdigest()[:24]
            )
            results[eid] = dict(
                evidence_id=eid,
                proposition_id=None,
                related_proposition_id=proposition_id or "entity:" + matched[0],
                role="mechanistic_context",
                direction="context",
                source_id=sid,
                publication_id=primary[0] if len(primary) == 1 else None,
                study_id=None,
                cohort_id=None,
                source_of_source=None,
                primary_publication_ids=primary,
                primary_knowledge_sources=references,
                aggregator_knowledge_source="UniProtKB",
                experimental_lineage="UNKNOWN",
                conditions={"species": record["organism"]},
                measurement_type="curation",
                source_locator=record["source_manifest"]["url"] + "#" + annotation["locator"],
                source_sha256=record["source_manifest"]["sha256"],
                resource_release=record["resource_release"],
                source_version=record["entry_version"],
                annotation_date=record["last_annotation_update"],
                text=annotation["text"],
                matched_entities=matched,
                scope_match="related",
                answers_question=False,
                integration_mode="CONTEXT_ONLY_OR_UNRESOLVED",
                license=record["license"],
                license_url=record["license_url"],
                attribution=record["attribution"],
            )
    return [results[key] for key in sorted(results)]
