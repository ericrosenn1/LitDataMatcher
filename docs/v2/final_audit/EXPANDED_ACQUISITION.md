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

Execution status and final receipts will be recorded here after acquisition.
