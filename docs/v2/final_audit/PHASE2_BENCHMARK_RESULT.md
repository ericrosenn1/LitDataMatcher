# Phase 2 real metadata benchmark result

The predeclared engineering and source-metadata evaluation both **PASS** on 1,600 unique literature records and 552 unique catalog accessions. This is `COMPLETE_VALIDATED` evidence for the bounded scale, retrieval, compatibility and recovery workload. It does not establish expert validation, calibrated probabilities, independent biological units, or completion of every final-campaign requirement.

The benchmark/protocol was committed before measurement as `cf2cf29`. The passing run used source commit `2c51a8c412c574b3a20692e18a671ab338d1fb88`, incorporating the lead's taxonomy repair `7bbc432`. Benchmark selection, reference labels, bounds, weights, implementation and protocol remained unchanged between runs.

## Executed evidence

All paths below are under `C:\Codex\LitDataMatcher-v2\data\final_campaign_20260913` and remain outside Git.

| Evidence | Exact relative path | SHA-256 |
| --- | --- | --- |
| Passing native receipt | `scale/run_20260913T220705Z/BENCHMARK_RECEIPT.json` | `28b389a5a8dcaca54f81f69fbd0618f79c727452f1843cfe6a65ed23e72ecfd7` |
| Preserved initial failure | `scale/run_20260913T220110Z/BENCHMARK_RECEIPT.json` | `fd7a88a59ec2ce4dc0176cec1793394d1c22d5c4264db5516d51fc4e22059e7d` |
| Literature input | `expanded/corpus/literature.jsonl` | `89fa856f6499d5dad9bb7a3a8b665c747a173b79214b607ea7e5bd3a713919a3` |
| Expanded catalog input | `expanded/corpus/datasets.jsonl` | `b70658adbaf9c0f926e8dcc96b99c3191db83d0d83008140b499453df94c1452` |
| Qualified omics input | `omics_qualification/studies.jsonl` | `d86ae76e37bbc3ef314fde3f7f56f8a89dba85f7aa44968ae38f5a7c7b8b9eff` |

The native receipt records the actual command, source-file fingerprints, hardware, local model manifest, input/acquisition receipt hashes, frozen plan hash, full measurements and 15 artifact hashes. The passing run contains `BENCHMARK_PLAN.json`, `QUERY_MANIFEST.json`, `RANKING_EVALUATION.json`, per-size `POINT_RECEIPT.json` files, clean SQLite catalogs, and both recovery child receipts/logs. `scale/validation/FINAL_BENCHMARK_VERIFICATION.json` independently verifies both runs' artifact hashes and 16 final comparison checks.

The catalog comprises 395 ClinicalTrials.gov, 75 MGnify, 17 ENA/SRA, 25 PRIDE and 40 Metabolomics Workbench accessions. All three inputs contain zero repeated identities. The merged catalog is a derivative of those input bytes; no synthetic replication contributes to these counts. Accession uniqueness does not establish participant, donor, cohort or study independence.

## Scale measurements

Execution used Python 3.12 on the local 32-logical-CPU workstation. The existing MiniLM encoder ran on CPU with four threads, batch size 16 and a 256-token limit. No model download or GPU execution occurred. Timings below are observed local measurements, not hardware-independent guarantees.

| Literature / catalog IDs | Ingest + FTS (s) | FTS optimize (s) | Warm query p50 / p95 (ms) | Matching s / 1,000 assessments | Context compilation (s) | Catalog disk (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 / 32 | 0.172 | 0.00155 | 0.041 / 0.417 | 0.0485 | 0.00107 | 3.438 |
| 500 / 100 | 0.917 | 0.00539 | 0.061 / 1.687 | 0.0352 | 0.00703 | 13.926 |
| 1,600 / 552 | 9.450 | 0.02761 | 0.220 / 21.653 | 0.0523 | 0.03496 | 49.832 |

Each size point includes 920 timed FTS queries. Matching includes source-record normalization and 17 questions against each actual candidate subset; the full point contains 9,384 assessments. Normalizing throughput to 1,000 assessments does not imply 1,000 unique datasets. Compilation uses one context-only item per actual literature record and preserves `independent_support_count = null` and `insufficient-coverage`; these are not extracted biological claims.

All points satisfy the predeclared query p95 <= 5 seconds and matching <= 30 seconds per 1,000 assessments. Sampled peak process-tree RSS was 1,247,211,520 bytes (1.162 GiB), below 8 GiB; 20-ms sampling can miss transient peaks. Catalog cache replay was 100% exact at every size, with zero misses or reinsertions. Query, catalog payload and compiler replay were exact. No network connection attempts were recorded. Loading the already qualified MiniLM took 12.36 seconds; encoding all 552 metadata entries took 18.38 seconds. Its vector SHA-256 was identical across both actual runs.

## Source-derived ranking evaluation

Seventeen source-anchored questions cover clinical registry, genomic, proteomic and metabolomic metadata, five explicitly recognized species, and observed joint requirements. Each method ranks the same 552 candidates exactly once: 9,384 question/candidate pairs comprise 1,376 observed fits, 6,439 observed mismatches and 1,569 unknowns. There are 4,114 observed organism-field mismatches and 4,463 modality-field mismatches; these categories overlap and must not be added as unique cases. Every label records `label_origin = source_determined`, its raw record hash, source locator and field-specific reason.

| Method | Mean judged P@5 | Mean R@10 | MRR | Mean known-label nDCG@5 | Confirmed-invalid top / 17 | Unknown top-5 slots / 85 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FTS lexical | 0.6941 | 0.5655 | 0.8838 | 0.8435 | 2 | 15 |
| Pretrained MiniLM | 0.6755 | 0.4865 | 0.8300 | 0.7948 | 3 | 19 |
| Weighted heuristic, eligibility ordering removed | 0.8471 | 0.6674 | 1.0000 | 1.0000 | 0 | 12 |
| Compatibility only | 1.0000 | 0.6674 | 1.0000 | 1.0000 | 0 | 25 |
| Compatibility + lexical | 1.0000 | 0.6674 | 1.0000 | 1.0000 | 0 | 25 |
| Compatibility + MiniLM | 1.0000 | 0.6674 | 1.0000 | 1.0000 | 0 | 25 |

Means are across all 17 queries. The pooled top-five fit/judged counts are respectively 47/70, 42/66, 60/73 and 60/60 for each compatibility variant; these pooled ratios differ from query means. Unknowns remain visible in full rankings and are excluded from judged-label denominators. Known-label nDCG explicitly removes unjudged positions; it is not an all-candidate accuracy measure. Per-query recall numerators and observed-positive denominators are retained in `RANKING_EVALUATION.json`.

Reference unknowns mean unjudged by the frozen literal table, which is deliberately narrower than all source terminology. Of 1,569 such pairs, the product returns inspection/unknown for 1,552 and incompatible for 17: six involve the extra full-name fruit-fly alias in `PXD073801`, and eleven involve the mixed `WGS`/`AMPLICON` source list in `PRJDB2729`. The benchmark retains all 17 as unjudged; it does not silently credit these product decisions as correct negatives. This evaluation does not establish exhaustive agreement for unmapped or partially mapped source vocabulary.

The compatibility path has zero observed mismatches promoted to qualified, zero observed fits excluded and zero unknowns promoted to qualified. Reversing candidate input order preserves exact ranking output. No semantic advantage is demonstrated on this narrow suite: the compatibility variants tie, and raw MiniLM retrieval is weaker than lexical retrieval on these metadata questions. Weights were not fitted. These queries derive from the evaluated catalog and include source-anchor self-rediscovery, so their scores do not estimate held-out scientific answerability or population accuracy. Status remains `UNCALIBRATED_HEURISTIC` and `PENDING_EXPERT_REVIEW`.

## Actual interruption, bounded repair and validation

Child process 37732 committed 276 catalog-record partitions and exited with code 71 without closing SQLite. A different child, 43760, reopened that catalog, validated the existing payloads, skipped those exact 276 identities and inserted only the remaining 276 before exiting 0. Final IDs, payload digest and the clean-catalog digest match; the database has exactly 552 current records and 552 versions, with no duplicates. Both executed commands, process IDs, receipts and hashes are recorded.

The first real run passed engineering but failed metadata qualification for three source records: `ST000001` (Arabidopsis thaliana), `PXD073801` (Drosophila melanogaster) and `ST003733` (Rattus norvegicus). Each failed species-only and joint-assay queries, producing six observed-fit exclusions. The lead repaired exact, taxonomy-qualified mappings and preserved unknown-organism semantics; see `TAXONOMY_QUALIFICATION.md`. The second run passed without changing any reference label, source record, candidate universe, query, benchmark bound or ranking weight. The original failure remains intact. Postvalidation confirms identical acquisition receipts, input bytes, query manifests, reference labels, clean catalog/compiler results, MiniLM vectors and unaffected retrieval baselines across both runs.

Validation: 22 focused tests passed (`tests/test_phase2_benchmark.py`, `tests/test_omics_adapters.py`, `tests/test_modality_contract.py`); benchmark Ruff checks passed. The seven benchmark tests use explicitly synthetic infrastructure fixtures, including actual child-process recovery, input tampering, duplicate/conflicting identities, unknown labels and fail-closed missing challenge coverage. Their fixtures are excluded from the real corpus. JUnit evidence is `scale/validation/benchmark_taxonomy_repair_tests.xml`. All 15 artifact hashes in each real run and all current input/source fingerprints verify. The original frozen alpha and sealed holdout were neither rerun nor modified by this benchmark.
