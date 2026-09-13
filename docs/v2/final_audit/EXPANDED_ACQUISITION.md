# Expanded Phase 2 metadata acquisition

Predeclared bounded acquisition, before live requests: four new domains
(cancer interventions, environmental microbiomes, metabolic interventions,
neurologic disease). Targets: at least 1,000 unique literature identities,
300 unique study IDs, two dataset repositories, and three new domains with
both literature and study metadata. These are acquisition counts, not
scientific validation or independent-cohort counts.

`scripts/v2/acquire_phase2_campaign.py` fixes exact queries and caps before
execution. Europe PMC receives four queries, at most 400 records each, dated
2018–2025. ClinicalTrials receives lung-cancer/immunotherapy,
breast-cancer/treatment, type-2-diabetes/intervention, and Parkinson queries,
at most 100 studies each. MGnify receives soil/marine/wastewater queries,
at most 25 studies each. ENA receives soil/marine metagenomic study-title
queries, at most 100 technical-run metadata records each, grouped into
studies by the existing adapter. No raw data URL is followed.

All adapters are existing project implementations. The existing immutable
`SnapshotClient` preserves actual HTTP response bodies, SHA-256 objects,
request parameters, retrieval times, and upstream status. A small interface
bridge provides the cache metadata required by the adapters. Execution is
serial, capped at 8 MB per response and three attempts per request, with
bounded provider-requested retry waits. No separately billed API is used.

Before requests, `PREDECLARED_PLAN.json` records source commit, script hash,
queries, bounds, resource profile, and protected-input hashes. Only immutable
holdout manifests and identifier/split reservations are read. Known frozen,
retired, contaminated, transfer, and development family IDs are excluded;
known protected PubMed/PMC IDs are also excluded in Europe PMC queries.
The sealed final-holdout manifest must retain its frozen SHA-256. No holdout
evaluation, label, rank, score, or scientific output is executed or tuned.
Undisclosed family links remain unresolved.

Data root:
`C:\Codex\LitDataMatcher-v2\data\final_campaign_20260913\expanded`.
Each successful `partitions/<id>/records.jsonl` has a hash-bound receipt;
failed attempts retain diagnostics and snapshots. Successful partitions are
reused without reacquisition. Raw snapshots and normalized partition variants
remain outside Git. The unique source corpus is `corpus/literature.jsonl` and
`corpus/datasets.jsonl`; per-domain copies are under `domains/<domain>/`.
First occurrence per exact DOI (literature) or source/study ID is retained,
with all original variants and partition memberships preserved separately.
Root owns downstream V2 normalization and analysis.

Offline replay denies request/socket networking and compares every acquired
partition after recursively ignoring only the known `cache_status` field.
All remaining record content must agree. Coverage, truncation, source errors,
counts, exclusions, and hashes remain explicit in acquisition receipts.
Bounded samples never imply an exhaustive source universe.

All 13 live partitions returned successful bounded responses. A technical
exclusion-filter error initially matched protected IDs embedded in negative
search queries inside provenance. The correction excludes provenance/request
bookkeeping from record-ID matching. Original attempt receipts and raw caches
are preserved; only the four literature partitions are reprocessed offline.
Queries, bounds, protected IDs, and dataset outputs remain unchanged.

Final bounded status: **PASS_WITH_LIMITATIONS**. All count targets are met;
the source universe remains a bounded sample, and downstream scientific
validation remains separate.

| Domain | Unique literature records | Unique study IDs |
| --- | ---: | ---: |
| Cancer interventions | 400 | 195 |
| Environmental microbiomes | 400 | 92 |
| Metabolic interventions | 400 | 100 |
| Neurologic disease | 400 | 100 |
| Total | 1,600 | 487 |

Study sources: **395 ClinicalTrials.gov**, **75 MGnify**, and **17 ENA/SRA**
IDs. Five duplicate trial records across the two cancer queries contribute
once to the corpus. The 395 distinct NCT IDs alone exceed the study-count
target. Repository aliases are preserved; study identifiers are not a claim
of independent cohorts. Abstracts are present for 1,456 literature records
and missing for 144; missing abstracts remain empty/metadata-only.

Validation: all 13 partitions replayed with networking denied, **0 network
guard calls**, and equal bytes after ignoring only `cache_status` recursively.
A guarded resume reused 13/13 partitions with unchanged request and corpus
hashes. The self-audit passed **15/15** checks. All 37 raw response objects
(20,183,548 bytes) passed SHA-256 verification. Every corpus ID maps to an
actual source row through `RECORD_SOURCE_LINEAGE.jsonl` (2,275 JSON-pointer
locators, including technical-run rows). All requests used the four declared
metadata APIs. Frozen manifest and identifier-file hashes remain unchanged.

Source requests ran from 2026-09-13 21:44:17 UTC to approximately 21:45:09 UTC.
Queries/bounds were persisted at 21:44:02 UTC, under acquisition commit
`18ee5b8b6c138795ebf82e3826f9e05fea48d39c`. The filter correction was committed
as `e0cab42` and completed from immutable caches, with zero network calls.
Original unsuccessful-filter attempt evidence remains under partition
`attempts/` and in `acquisition_run1.log`.

Final source files and SHA-256:

- `corpus/literature.jsonl`:
  `89fa856f6499d5dad9bb7a3a8b665c747a173b79214b607ea7e5bd3a713919a3`
- `corpus/datasets.jsonl`:
  `b70658adbaf9c0f926e8dcc96b99c3191db83d0d83008140b499453df94c1452`

Receipts: `ACQUISITION_RECEIPT.json`, `OFFLINE_REPLAY.json`,
`RESUME_VALIDATION.json`, `SELF_AUDIT.json`, `EXCLUSION_REPAIR.json`,
`SOURCE_COVERAGE.json`, `DATA_MANIFEST.tsv`, and per-partition receipts.
The compact handoff includes code, receipts, plan, lineage, and a data manifest;
raw and normalized data stay at the external data root.

The inherited Europe PMC adapter attaches the final response's provenance
metadata to every row while retaining the correct per-page request list.
`RECORD_SOURCE_LINEAGE.jsonl` supplies the exact record-to-response mapping.
Lead-owned adapter/normalization repair can consume this mapping without
reacquisition. This known downstream correction is not silently treated as
validated row-level adapter provenance. No core adapters, normalization,
scientific thresholds, frozen outputs, or holdout evaluation were changed by
this worker.
