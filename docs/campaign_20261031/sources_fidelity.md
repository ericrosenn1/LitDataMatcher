# Source fidelity changes

This workstream starts from integrated incumbent `e592407`. It repairs source interpretation and retrieval without changing ranking weights, sample-size thresholds, extraction rules, or protected evaluation inputs.

## Versioned interpretation

`source-evidence-v2` distinguishes source-reported metadata from inferred and absent capability fields. A positive, explicitly supplied metadata field can produce an `observed` capability: observed in metadata, not measured or analysis-ready. Zero-completeness fields are `ABSENT`; text-inferred and proxy fields are `INFERRED`. These remain visible as `unknown` capabilities and cannot supply derived-capability prerequisites. Derivation rules and their confidence values are unchanged.

ClinicalTrials retains registry-field support heuristics. An empty eligibility module is no longer treated as present because an empty dictionary was wrapped in a list. Enrollment stays registry metadata rather than analyzed sample size.

`metadata-aware-v2` is the default governance policy. It uses existing numerical levels but gives explicit restrictions precedence over public-metadata terms. Public metadata does not establish underlying-data availability. `assess_governance(record, policy="legacy-v1")` preserves the prior policy for historical comparisons. Results record `policy_version`. Downstream rankings can change through this declared interpretation repair; the ranking weights remain unchanged.

## Acquisition lineage

New cache writes produce immutable receipts in `.provenance/`, keyed by the unchanged request-cache filename and response-content SHA-256. The receipt is atomically created before promoting the payload. A payload's digest selects its receipt, preventing an interrupted refresh from attaching new acquisition metadata to old content. Identical content retains its first recorded acquisition time. Cache hits change `cache_status` while preserving that time despite filesystem timestamp changes.

The digest identifies the cached representation, which may be normalized JSON; it is not a hash of original HTTP wire bytes. Receipts retain request scope with common credential parameters redacted. Complete immutable receipts can survive failed payload promotion; they do not count as successful cache hits. Invalid receipts fail closed.

Legacy cache files remain readable without modification. Their timestamp basis is explicitly `legacy_cache_mtime_unverified`. Copy `.provenance/` with payloads to preserve new acquisition evidence. Cache writes require same-volume atomic replacement and hard-link support.

OpenAlex, GEO, and PubMed now retain response acquisition lineage rather than substituting normalization time. Existing digests and per-page provenance remain available.

## Source contracts

GEO validates ESummary envelopes and accession records. Errors cannot create fallback datasets. Valid records survive individual missing or malformed records with partial/failure status and missing IDs.

The optional `datacite` source filters for datasets, validates DOI identity, retains source rights and related identifiers, and follows up to four cursor pages of 25 records. Pagination only accepts the expected HTTPS DataCite endpoint. Malformed records and pagination produce partial status while retaining valid records.

DOI metadata does not establish file access or measured variables. DataCite sample count remains unknown, represented by the legacy zero sentinel plus explicit null/status metadata. Text-derived suggestions remain inferred. Source-specific terms require review. See the official [query documentation](https://support.datacite.org/docs/api-queries) and [pagination contract](https://support.datacite.org/docs/pagination).

## Remaining boundaries

- Rankings remain uncalibrated. Legacy direct variable-alias matching is unchanged; this capability repair does not rewrite previous outputs.
- The HTTP client carries mutable response metadata. Concurrent tasks should use separate client instances.
- Source pagination is bounded; completing a query does not establish a scientific census.
- HTTP retry/backoff retains the incumbent policy.
- No corpus, model, sealed-holdout, deployment, or protected-output changes were made.
