# Source fidelity validation — 2026-10-31

Ten initial regression cases reproduced nine incumbent failures; the existing positive derivation case passed. After repairs, the first combined source/cache/omics round passed 93 cases. A second expanded round passed 128 cases, including provenance, review exports, and provenance graph checks. Ruff passed for the changed modules and tests.

Evidence is separate from source and protected data:

- `C:\Codex\LitDataMatcher-v2\data\campaign_20261031\sources\validation_20261031T115332Z\VALIDATION.json`
- `C:\Codex\LitDataMatcher-v2\data\campaign_20261031\sources\validation_20261031T115332Z\targeted_round_2.xml`
- `C:\Codex\LitDataMatcher-v2\data\campaign_20261031\sources\metadata_probe_20261031T115314Z\PROBE_REPORT.json`
- `C:\Codex\LitDataMatcher-v2\data\campaign_20261031\sources\validation_final_20261031T115418Z\VALIDATION.json` — final 128-case check and lint after registering the DataCite interpretation profile.

## Covered contracts

- Provider errors and missing/malformed GEO records cannot create synthetic datasets; valid partial results survive.
- Cached acquisition time and content digests survive normalization and filesystem timestamp changes.
- Missing/inferred/proxy fields cannot become observed capabilities or derivation prerequisites; explicitly supplied positive metadata still supports existing derivation rules.
- Mixed public-metadata/restricted-data governance has an explicit policy version and callable legacy policy.
- Cache receipts reject mismatched digests, retain first acquisition for identical content, redact credential parameters, and preserve prior payloads on failed promotion.
- DataCite validates dataset type/DOI identity, retains rights and publication relations, deduplicates DOIs, and reports unsafe/repeated pagination.

## Bounded live checks

Only public metadata was requested. Raw-response caches, immutable acquisition receipts, normalized records, and a hashed inventory were retained in the probe directory.

| Source | Metadata records | Fresh offline replay |
| --- | ---: | --- |
| OpenAlex | 2 | Equal except intentional cache-status transition |
| GEO | 10 | Equal except intentional cache-status transition |
| DataCite | 1 | Equal except intentional cache-status transition |

The DataCite probe used one discovery result and then an exact DOI query. GEO returned several accession-linked records, not necessarily independent studies or participants. No matrices, biological raw data, or model weights were downloaded.

These checks establish bounded software and retrieval behavior, not scientific suitability, source completeness, independent samples, or calibrated scores.

## Reproduce the focused checks

```bash
python -m pytest tests/test_source_fidelity.py tests/test_datacite_adapter.py tests/test_http_cache.py tests/test_adapters_and_capabilities.py tests/test_omics_adapters.py tests/test_provenance_interpretation.py tests/test_review_source_metadata.py tests/test_provenance_dag.py tests/test_deep_nodes.py
```

The lead's integrated validation is separate. This workstream did not duplicate the full suite or previous scientific runs.
