# Bounded 100,000-abstract rules continuation

The completed demonstration retains **100,000 distinct native PubMed records**
in 100 batches of 1,000 nonempty abstracts. It imports all ten previously
completed batches, preserving their exact input bytes and extraction payloads,
and extracts only the additional 90 batches. No model inference, external
retrieval, dataset inspection, qualification, or global grouping was performed.
The original 10,000-record and global-grouping archives remain frozen.

## Measured result

| Accounting | Count |
| --- | ---: |
| Native nonempty abstract inputs | 100,000 |
| Processed source views | 99,909 |
| Excluded by existing lifecycle policy | 91 |
| Processed views without a configured cue (`EMPTY`) | 92,759 |
| Views with retained passages (`SUCCESS`) | 7,150 |
| Exact retained passages | 7,883 |
| Source-stated unresolved passages | 5,549 |
| Source-stated limitations | 1,659 |
| Source research questions | 675 |
| Input null values preserved | 114,517 |
| Result null values retained | 168,759 |

There were no extraction failures or truncation. All retained passages passed
literal text/offset, text hash, native ID, citation, locator, provenance, and
existing lifecycle checks. Null counts describe different input and result
schemas; they do not imply every input field occurs in each question record.
`EMPTY` means no configured cue, not absence of a research gap.

Preparation took **26.756 seconds**. New rules extraction plus grouping within
each batch took **44.153 seconds** across the 90 new batches; historical
extraction times were retained but excluded from this total. Maximum sampled
process RSS during new extraction was **84,525,056 bytes (80.61 MiB)**, sampled
every 20 ms. This is not the peak for preparation or the entire application.

The planned-stop invocation took 4.940 seconds; the resumed invocation including
archive creation and verification took 79.463 seconds. Including the initial
40.168-second read-only feasibility count, these measured calls total 151.327
seconds, below the ten-minute compute target. Development/test work and the
separate independent artifact audit are outside that timing.

## Sampling and recovery boundaries

The original ten batches retain committed snapshot prefixes 677, 885, 886, and
887. The added cohort continues below their minimum PMID, 19,795,460, using the
same exact-phrase breast-oncology query, explicit topic regex, and pathogen
wording exclusion. A bounded candidate-ID list was frozen once; source views
were then copied in short read-only transactions, all at prefix 1,353. The ID
selection snapshot and each source-view snapshot remain recorded separately.
Later corpus materialization does not change these frozen views.

This is a mixed historical cohort, not one corpus snapshot or a representative
sample. Grouping remains bounded to each 1,000-document batch. The 7,875 groups
summed across batches are not a globally deduplicated group count. The earlier
935-passage global-grouping demonstration remains separate.

The native `CheckpointRun` stopped after batch 12: ten imported batches and two
new extractions. Resume loaded all twelve without extraction or rewriting their
successful checkpoint files; hashes and modification times were unchanged.
The final run records 100 successful extraction nodes. Core extraction and
scientific dependency hashes matched the prior run; changes confined to four
execution/application modules were explicitly recorded in the import proof.
No scientific rule or threshold changed.

## Artifacts and validation

Runner: `scripts/validate_question_discovery_batches.py`, commit `55c73cd`.
The new `prepare --prior-run PATH --count 100000 --max-seconds 300` mode imports
verified completed native checkpoints and prepares the bounded continuation.
Existing `run --stop-after 12` and `run --resume` demonstrate native recovery.
Eight focused script regressions pass, including multi-batch native import
without recomputation, changed-artifact rejection, exact count reconciliation,
and resume/dependency safeguards.

The measured receipt is
[`question-discovery-100000-scale-20261031.json`](../validation/question-discovery-100000-scale-20261031.json).
Inputs are under
`C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\question-scale-100000-inputs-v1`;
outputs and the adjacent verified archive are under
`C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\question-scale-100000-v1`.
The ZIP contains exact inputs, native results/checkpoints, source-selection and
prior-run proofs, summaries, validation, runner code, and a checked SHA-256 manifest.

Archive: `question-scale-100000-v1.zip`, 108,811,618 bytes.
SHA-256: `3e1e58dc20c852eacd948c965e0d4142ee44ea6c9b03733cf882aebf9b312f1d`.
Every archived member matched its manifest size and hash; ZIP CRC and membership
checks passed. No further extraction run is needed.

An independent read-only artifact review reconciled all 100,000 identities and
7,883 question IDs/spans, status/index/retained counts, lifecycle decisions,
nulls, all ten imported result payloads, and twelve resumed checkpoint files.
It verified all 215 ZIP members against disk and rehashed 42 earlier files
(95,538,270 bytes), including both earlier archives: all unchanged. It also
independently reran the eight focused tests. No substantive finding remained.

These results establish bounded engineering execution, exact-source fidelity,
and checkpoint reuse. They do not establish reviewed extraction accuracy,
global novelty, independent evidence, statistical adequacy, or a 100,000-record
end-to-end model/matching benchmark.
