# Source-link retrieval experiment protocol

This is a new development experiment, not a re-evaluation of the sealed holdout. Its task is to recover a question's originally linked paper from a fixed source-paper candidate pool. It does not label other papers scientifically irrelevant, assess dataset feasibility, or create gold data-requirement annotations.

## Inputs and exclusions

- Use only the existing `train` and `development` allocations after corpus-file integrity qualification.
- Exclude the `evaluation_candidate` allocation and all protected final-evaluation content.
- For Evidence Inference, intersect training with official train article IDs and development with official validation article IDs.
- Exclude EvidenceBench's official test data and unidentified paper IDs. Do not coerce a numeric source-local identifier into a PMID.
- Retain one deterministically sampled question per source paper. Resolve PMID/PMCID aliases and reject cross-split paper and exact-question overlap.
- Use the independently qualified 2026 annual PubMed baseline. Incremental updates are excluded until their version/deletion semantics are independently qualified.
- Missing articles or abstracts and retracted publication types are counted as exclusions, never fabricated or replaced without disclosure.

## Fixed candidates

1. BM25 (`k1=1.2`, `b=0.75`).
2. Pretrained all-MiniLM-L6-v2, immutable revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, masked mean pooling and normalized vectors.
3. BM25 plus pretrained embeddings through reciprocal-rank fusion (`constant=60`).
4. At most two supervised local MiniLM fine-tuning epochs using only known training links and in-batch contrastive negatives. Defaults: batch 16, learning rate 0.00002, weight decay 0.01, temperature 0.05, 256-token limit, seed 1729. Other papers are unjudged negatives, a reported limitation.

The pretrained weights remain immutable. Each trained epoch is a new model directory with weight hashes, parent revision, input hashes, runtime versions, and a distinct training fingerprint. No live weight updates occur during ordinary discovery.

## Evaluation and promotion

Report per-query source-link recall@1/10, MRR@10, nDCG@10, paired query-level changes, descriptive bootstrap intervals, exclusions, source strata, token truncation policy, and cold/warm indexing time. Empty rankings contribute zero, and denominator changes invalidate paired comparisons.

For this development experiment only, a trained candidate is eligible for an **experimental** promotion if MRR@10 improves by at least 0.01 and recall@10 falls by no more than 0.005. If no candidate passes, retain the pretrained incumbent and report the negative result. This rule is recorded before training; it does not replace any historical acceptance threshold. Development selection is not independent test validation, and neither scores nor intervals establish scientific truth.

Completed epochs and their optimizer checkpoints are retained. Identical interrupted runs may resume after fingerprint verification; changed data/configuration/model requires a new run. A retry must not overwrite a validated prior model or silently change batch size, precision, or input scope.
