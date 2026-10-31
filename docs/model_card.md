# Model and evaluation card

Upstream model, dataset, and software attribution is collected in the
[primary references](campaign_20261031/references.md).

LitDataMatcher supports research triage: source statements, typed questions,
and candidate datasets for inspection. Its outputs do not establish scientific
truth, causal identification, clinical utility, or access to participant data.

## Components

| Component | Model or method | Role |
| --- | --- | --- |
| Local extraction | Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28` | Source-bound claims and explicit future-work statements |
| Smaller candidate | Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Not recommended after failing both literal-profile positive smoke cases |
| Pretrained encoder | all-MiniLM-L6-v2, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Candidate retrieval |
| Experimental encoder | Two-epoch adaptation of pinned MiniLM | Original-paper link recovery, evaluated separately |
| Lexical comparator | BM25, `k1=1.2`, `b=0.75` | Simple retrieval baseline |
| Fusion | Reciprocal rank fusion, constant 60 | Combined lexical/semantic retrieval |

The Qwen extractors are not trained by this project. Manifests retain revision,
tokenizer, file hashes, and upstream license information. Runtime inference
does not download weights or execute remote model code.
The [model-file contract](campaign_20261031/model_asset_integrity.md) covers the
complete local inference asset set and validates referenced weight/configuration
files. This loader accepts full local checkpoints; adapter-only checkpoints
requiring an external base model are outside its supported contract.

## Retrieval adaptation

Training uses 1,333 source-linked questions; development uses 175. Each selected
paper contributes at most one query. The initial collection contains 1,508
papers from the qualified annual PubMed baseline. Evidence Inference and
PubMedQA partitions are checked against declared allocations. Official test
partitions and reserved project evaluation data are excluded.

Training uses two epochs, batches of 16, learning rate `2e-5`, weight decay
`0.01`, contrastive temperature `0.05`, and seed `1729`. Encoding is limited to
256 tokenizer tokens. In-batch alternatives are unjudged negatives, not
confirmed scientifically irrelevant papers.

The pretrained-relative selection rule was fixed before training: MRR@10 must
improve by at least 0.01 and recall@10 may decline by no more than 0.005.
Passing this rule is an experimental selection, not a production promotion or
a claim that the encoder beats BM25.

## Measurements and limitations

The initial experiment selected epoch 2 over its pretrained parent, but BM25
remained stronger overall. Many PubMedQA questions echo target titles. See the
[retrieval protocol](campaign_20261031/retrieval_protocol.md) and the repository's
`benchmarks/campaign_20261031/retrieval_development.json`. Larger-pool and
title-removal comparisons are separate development diagnostics. Both completed
large-pool comparisons retain BM25 as the strongest measured method. Fine-tuning
improves aggregate dense retrieval but reduces PubMedQA MRR in abstract-only
conditions. Complete results are recorded in
`benchmarks/campaign_20261031/retrieval_scale.json`.

Extraction qualification uses previously seen development documents. The
smaller model failed both literal-profile positive smoke cases and was not
advanced to the 30-case comparison. Qualification requires fresh
inference, source validation, and verified offline replay; it is not an
expert-labelled accuracy estimate. See the
[extraction protocol](campaign_20261031/extraction_protocol.md).

The constrained profile emits at most one claim per source window, copies exact
complete sentences, and uses null context/comparator fields. Native analysis
has explicit window/count limits and can omit evidence outside selected
passages. The independent verifier checks spans and several relation,
direction, and negation invariants; it cannot validate every scientific
interpretation. Accepted-record counts do not establish accuracy.
Subject, predicate, and object preserve surface sentence structure; a subject
can be a reporting word such as “Results.” These fields are not normalized
biological entities or validated causal triples.

No expert-labelled question–dataset suitability benchmark or calibrated
probability model is claimed. Paper-link retrieval does not validate matching
to dataset metadata. An adapted paper encoder is not automatically better for
that different task. Unknown capabilities remain unresolved in dossiers.

Paper-disjoint task training does not prove absence of overlap with upstream
model pretraining; that overlap is unmeasured. The extraction rules primarily
target English biomedical text. Multilingual performance is not established.

## Reproduction

Use pinned model manifests, recorded dataset hashes, and versioned protocols.
Training, scale evaluation, source qualification, and runtime checks have
separate scripts under `scripts/campaign_20261031/` and `scripts/v2/`. Reproduce
the recorded source revision before resuming its checkpoints; changed inputs
or code are rejected. Timings are local measurements, not guarantees for other
systems.
