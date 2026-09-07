# E04 source-snapshot matching evaluation

`benchmarks/v2/evaluate_matches.py` freezes 20 primary and 10 transfer source-order records from the acquired `studies.jsonl` catalogue before any rank result is read. It excludes the GSE112372, GSE214695, and GSE226875 connected families and rejects the run if the catalogue no longer reproduces the frozen selection.

Each query is a no-accession, source-described experimental-context retrieval task. The full within-split candidate universe is labelled: the exact source-described record has relevance 3 and every other candidate has relevance 0. These are source-determined metadata-retrieval labels, not expert biological-fit labels or truth probabilities. The output records all source locators and snapshot hashes, negatives, reported sample counts, and comparator unknowns. It never converts source-reported samples to independent donors.

The run compares lexical overlap, the local pretrained all-MiniLM-L6-v2 encoder on CPU with a 50/50 lexical/cosine hybrid, and the lead `scientific_v2.rank_candidates` compatibility gate on identical universes. It reports raw metric numerators and denominators, nDCG@5 and precision@5 descriptively, and zero-tolerance invalid-top counts.

## Executed result

`benchmarks/v2/E04_SOURCE_SNAPSHOT_MATCHING.json` was produced from catalog SHA256 `e52aef7e18d7c10b4a151af2c4ee005a4d92601a3ece53d11b70a6df8ffc7cc8` using `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` through the local Transformers CPU runtime, with no retrieval cache.

| Split / method | Recall@10 | P@5 | nDCG@5 | Invalid top |
|---|---:|---:|---:|---:|
| Primary 20 queries, lexical | 20/20 | 16/100 | 0.5651 | 13/20 |
| Primary 20 queries, MiniLM hybrid | 20/20 | 20/100 | 0.9381 | 3/20 |
| Primary 20 queries, compatibility-aware | 20/20 | 20/100 | 1.0000 | 0/20 |
| Transfer 10 queries, lexical | 10/10 | 10/50 | 0.8431 | 3/10 |
| Transfer 10 queries, MiniLM hybrid | 10/10 | 10/50 | 0.9631 | 1/10 |
| Transfer 10 queries, compatibility-aware | 10/10 | 10/50 | 1.0000 | 0/10 |

The primary capability audit checked 100 fields: organism, assay, source title, source-reported sample count, and retained unknown comparator for each of 20 source families. It found 100/100 as represented, 20 comparator unknowns retained, and zero inferred independent-donor counts. The metadata-retrieval pilot therefore meets its declared 95% field, 90% Recall@10, and zero invalid-top engineering floors. It does not meet the protocol's broader scientific-calibration, live/replay, linkage-closure, untouched-holdout, or product-closeout requirements.

GSE112372 detailed metadata was accidentally exposed in targeted acquisition. This evaluator creates no predictions or labels for it and does not claim that its primary holdout remains untouched. GSE214695 and GSE226875 are excluded from development and no transfer-holdout claim is made.

E03's 12 controller/reliability checks passed independently in `C:\Codex\LitDataMatcher-v2\data\evaluation\E03_controller_independent.xml`. That supports only the scheduler prerequisite verdict: the tested controller recovery/lease/path/repair contracts are ready for the next scheduled-supervisor prerequisite review. It is not a whole-product, scientific-calibration, or scheduled-operation approval.
