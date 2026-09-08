# Current continuation

Status: `RUNNING` / `HARDENED_ALPHA_COMPLETE`, current phase `PHASE_2`. Canonical lead: `C:\Codex\LitDataMatcher-v2\lead`; data/state: `C:\Codex\LitDataMatcher-v2\data`; branch: `codex/litdatamatcher-v2-build`.

The one-time v4 final holdout at `data\evaluation\final_holdout_v4\run` passed and is sealed: do not rerun it. Preserve completed acquisition, local model qualification, final integrated run, release archives, and deterministic test evidence. The corrective supervisor is paused and model/effort runtime-unverified, so it remains unavailable. All Codex continuation must be `gpt-5.6-terra` at `high` or below; the current lead is Terra Medium. Supported shared-account telemetry projects weekly exhaustion before reset at the observed rate, so use one Terra Medium reasoning worker only, with no concurrent reasoning workers; prefer deterministic local work and pause reasoning while long deterministic jobs run.

Frozen V2.0-HARDENED-ALPHA baseline: source commit `5747cbea2ae65c8570280d0e53f77bfabc968712`; final3 wheel/sdist and acceptance evidence are retained under `C:\Codex\LitDataMatcher-v2\data\releases\0.2.0-hardened-alpha-final3` and `C:\Codex\LitDataMatcher-v2\data\acceptance`. The first bounded V2.1-MULTISOURCE tranche is complete at source commit `3a3d734a5a6f2dd1580bcc7fa51d21515b08aa41`: optional Europe PMC and Crossref metadata adapters retain stable identifiers, provenance timestamps, cache snapshots, DOI-based cross-source relations, and fail-closed offline cache replay. Qualification is limited to one metadata record per source under `C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_multisource`; it did not invoke acquisition or a full-text download.

The ClinicalTrials.gov registry-metadata tranche is complete at source commit `1d230cefa4e2e0ff49f63dc7e672b60ea1a77e7f`. It preserves study status/type/version, conditions, interventions, comparators, outcomes/timepoints, eligibility, phase, enrollment unit, arms, access limits, explicit missingness, and cache provenance. Registry enrollment remains distinct from analyzed sample count, and observational records are `NOT_PERTURBATIONAL`. One-record live and byte-identical offline-cache replay evidence is under `C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_clinicaltrials`.

The ENA/SRA tranche is complete at source commit `772331f6cc3f5a1de57964b913f6a5a937ba7cd9`. It groups bounded run metadata under a study while retaining stable study/sample/run/experiment IDs, secondary accessions, availability, version time, provenance, typed missingness, and explicit run-to-sample/dependence links. Technical runs are never counted as biological samples and donor links remain ambiguous. Evidence is under `C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_ena`.

For the next V2.1 continuation, first verify the frozen baseline and replay the bounded source cache:

```powershell
C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe scripts\v2\validate_alpha_baseline.py --source C:\Codex\LitDataMatcher-v2\lead --data C:\Codex\LitDataMatcher-v2\data --out C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_clinicaltrials\alpha_baseline_non_regression.json
C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe -m litdatamatcher.cli dataset-search --query microbiome --source clinicaltrials --limit 1 --cache-dir C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_clinicaltrials\http_cache --offline --out C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_clinicaltrials\continuation_offline.jsonl
```

Do not restart acquisition, download application models, rebuild validated artifacts, rerun holdout scoring, or weaken scientific acceptance criteria.

## V2.3 evidence compiler checkpoint

The bounded evidence-compiler contract tranche is validated locally: exact shared study/cohort/publication/source identifiers and explicit source-of-source paths form auditable known-dependence edges; source-located assertions retain replication, orthogonal, direct perturbational, associative, mechanistic, indirect, contradictory, incompatible, and unknown-dependence classifications without inferring them from text. Only same-underlying, derivative, and duplicated-cohort edges join a known-dependence group. The deterministic receipt is under `C:\Codex\LitDataMatcher-v2\data\phase2\v2_3_evidence_compiler\contract_receipt.json`. Frozen alpha and sealed holdout evidence remain untouched.

## V2.4 expert-review checkpoint

The review machinery is `PENDING_EXPERT_REVIEW`, not calibrated or expert-validated. Versioned packets preserve source spans/provenance while masking ranking/model fields and reviewer identity. Strict categorical labels, descriptive agreement, and pending-adjudication records are ready for real expert input. The deterministic zero-label infrastructure receipt is `C:\Codex\LitDataMatcher-v2\data\phase2\v2_4_expert_review\packet_receipt.json`.

## V2.5 scale/recovery checkpoint

Bounded local scale instrumentation is validated at `C:\Codex\LitDataMatcher-v2\data\phase2\v2_5_scale\benchmark_receipt.json`. It measures only the declared synthetic fixture and records its host/backend/limit provenance, throughput/latency, memory/disk, cache replay, and reopen/resume. It performs no network acquisition, model inference, LLM context loading, or production-scale claim.

## Literature integrity checkpoint

Normalized multi-source literature metadata now retains lifecycle/version/retraction state, source snapshots/statuses, full-text unknowns, dedup lineage, and deterministic derivation invalidation. Retrieval/schema failures remain unknown. Corrected, retracted, versioned, or cross-source duplicate metadata cannot silently become evidence or independent support. The synthetic receipt is `C:\Codex\LitDataMatcher-v2\data\phase2\literature_integrity\receipt.json`.

## Requirement formalization checkpoint

Matching now exposes expanded machine-readable compatibility status while preserving existing eligibility: exact/directly answerable/partial/indirect/additional-data/incompatible/unknown. Field-level observations and provenance remain authoritative; absent metadata is not incompatibility.

## Entity normalization checkpoint

Local identifier contracts now preserve candidate sets and source/mapping state across major entity classes. Only unambiguous exact/synonym IDs participate in requirement matching; ambiguity, deprecation, orthology, unresolved values, and source failures remain unknown/reviewable.
