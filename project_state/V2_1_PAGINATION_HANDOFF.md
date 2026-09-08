# V2.1 adapter pagination/completeness contract slice

Objective: ensure bounded metadata pages from Phase 2 literature and repository adapters cannot be silently presented as a complete evidence or candidate universe.

Implementation: Europe PMC and ClinicalTrials.gov use a shared bounded cursor reader. Each retained page records URL and request parameters, cursor in/out, returned item count, and cache lineage. A terminal cursor is `COMPLETE`; repeated token, maximum-page truncation, schema drift, and fetch error are non-complete statuses with `PARTIAL_CANDIDATE_UNIVERSE_NOT_EVIDENCE_COMPLETE`. Europe PMC stores the contract in row provenance/metadata; ClinicalTrials stores it in record metadata.

Validation: `python -m pytest tests/test_adapters_and_capabilities.py tests/test_literature_and_ranking.py -q` passed (`37 passed`). Mocked tests cover a complete two-page sequence, repeated cursor, bounded truncation, schema drift, explicit fetch error, and two-page offline replay with per-page `cache_hit` lineage. The deterministic receipt is PASS at `C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_pagination\receipt.json`, SHA-256 `befcde8af9d843382d85a7e96f74db4403a2d98c68d13d5e07f2ecda74ebc0f5`.

Limitation: this is a local bounded pagination contract. It does not invoke any live endpoint, enumerate a source, prove external completeness, acquire datasets/full text, run models, or modify frozen alpha/holdout artifacts.
