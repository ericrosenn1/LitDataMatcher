# Extraction baseline: measured results and limits

The original Qwen2.5-7B `source_schema` run completed **30 development cases:
9 PASS, 21 PARTIAL, no FAILED cases, and 2 retryable cases**. These are workflow
statuses, not scientific correctness labels. The run retained **16
source-guard-accepted claims** and established **no scientifically qualified
dataset matches**.

The machine-readable [baseline record](../../benchmarks/campaign_20261031/extraction_baseline.json)
contains exact counts, configurations, identities, receipt hashes, and failed
follow-up qualifications. The [protocol](extraction_protocol.md) defines the
qualification rule and interpretation limits.

## Inputs and outputs

The case pack was frozen before runtime results: 15 source-linked Evidence
Inference 2.0 cases and 15 PubMedQA cases, with unique linked papers. It used
annual-baseline abstracts; update-stream records and complete article bodies
were excluded. These are development cases, without expert gold labels. Because
observed failures motivated later repairs, this pack is now development/regression
evidence and must not be presented as untouched final evaluation.

| Persisted output | Count | Interpretation |
|---|---:|---|
| Accepted claims | 16, across 14 cases | Passed the existing extractive source guard |
| Caller questions | 30 | Supplied input questions, not model discoveries |
| Source-extracted future-work questions | 5, across 4 cases | Exact source-derived questions; novelty unestablished |
| Dataset candidate rows | 140 | Every row requires inspection |
| Scientifically qualified candidate rows | 0 | No qualified fit established in any case |

Twenty questions across 17 cases had `NO_ASSESSABLE_CONTRACT`: 15 caller questions
and all five extracted questions. One caller question had
`NO_RELEVANT_CANDIDATE_FOUND`. The other 14 caller questions produced the 140
inspection-needed rows. Incomplete requirements and metadata are not evidence
that suitable datasets do not exist.

## Source guard versus technical failures

Of 45 source views, 43 returned runtime records: **24 abstained**, 14 were
validated, and 5 were validated with rejections. Two views failed technically:
one exhausted two attempts on source-quote enumeration, and one encountered an
unsatisfiable token-prefix constraint before producing raw output.

The 73 preserved completed generations comprise 54 with a source-guard
rejection, 16 with accepted records and no rejection, one valid empty output,
and two schema rejections. Six rejection-containing generations also retained
valid records; rejection-containing output is not equivalent to full abstention.
Retry-level outputs are not the final deduplicated claim/question counts.

Attempt-level rejected records were: unsupported direction 25, non-verbatim
predicate 7, nonunique/overlapping relation spans 6, non-verbatim object 5,
cross-clause relation 4, contradictory negation 4, non-verbatim subject 2, and
direction attached to another predicate 1. The final manifests separately record
26 source-guard events and two inference failures. These counting units are
kept distinct in the JSON record. Transport repair alone does not resolve the
relation, direction, or negation failures.

## Follow-up qualification: source-index remains unpromoted

The same two previously seen positive development documents were used under
the existing rule: **both must yield at least one accepted claim and pass
offline replay**. They are qualification inputs, not independent evaluation.

| Model / execution | Document | Qualification | Accepted claims | Extracted questions | Fresh seconds |
|---|---:|---|---:|---:|---:|
| Qwen2.5-1.5B, CPU / float32 | 1 | FAIL | 0 | 0 | 112.88 |
| Qwen2.5-1.5B, CPU / float32 | 2 | FAIL | 0 | 0 | 25.94 |
| Qwen2.5-7B, CUDA / bfloat16 | 1 | PASS | 1 | 0 | 29.24 |
| Qwen2.5-7B, CUDA / bfloat16 | 2 | FAIL | 0 | 1 | 44.45 |

The smaller model produced direction-guard rejections on document 1 and valid
empty arrays on document 2. The 7B document-2 claim failed the negation guard;
its extracted question does not satisfy the required-claim rule. Thus both
source-index model configurations remain **unpromoted**. CPU and GPU timings
are not a controlled model-speed comparison.

**Source-literal follow-up: PENDING.** This baseline record asserts no result
from that separate qualification. Its synthetic transport checks are described
in the [profile documentation](model_source_literal.md).

## Timing and reproducibility

The baseline ran on 2026-10-31, 12:33:57–12:51:17 UTC: **1,040.48 seconds
(17.34 minutes)**. Median case-attempt time was 28.80 seconds. Returned runtime
views had median 22.54 seconds; the two technical failures have no returned
view duration. CPU retrieval scoring and a two-process corpus ledger ran
concurrently. These single-machine observations are not throughput guarantees.

All 43 runtime records share one process ID, and all 30 retrieval receipts are
identical. Together with the shared-session implementation, this supports model
and index reuse. No weight-load counter or captured loading stderr establishes
an exact count of model loads. The network evidence is a Python socket/DNS audit
hook with a blocked probe and no subsequent blocked events, not an OS firewall
or packet-capture claim.

The sealed report checked 480 original output receipts, 73 raw copies, source
offsets, and inference fingerprints. Full SHA-256 values are in the JSON record,
including the case manifest and these two anchors:

- Frozen pipeline input: `9b259ae42cd8b1eb8c71890d6674d53e46bb2fc3cad29d6225a69a69cd186024`.
- Closed batch state: `fef64d3a048b89ccd5d8c8be8331c40920fe8df38a5aa8a9f80d6e623366ba38`.

Raw source text is excluded from this publication summary. Passing software and
source-span checks does not establish scientific truth, novelty, extraction
accuracy, or calibrated dataset suitability.
