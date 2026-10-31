# Local extraction qualification and size comparison

This is a software/runtime development experiment, not a scientific accuracy
benchmark. No expert labels or untouched evaluation results are available.

## Fixed inputs and configuration

- Smoke inputs: the two pre-existing development qualification documents,
  `qualification_document.json` and `qualification_document2.json`. These were
  already used for prompt development; they are not independent evaluation.
- Main development pack: the 30 source-linked cases frozen by
  `build_research_cases.py` before inspecting their model results.
- Candidates: the already installed, immutable Qwen2.5-7B-Instruct and
  Qwen2.5-1.5B-Instruct snapshots. No new model training or downloading is needed.
- Both candidates use `source_schema`, 384 output tokens, at most two attempts,
  seed 1729, CUDA/bfloat16, and the same source guard and compiler.
- Keep the failed unrestricted 7B and initial constrained 7B attempts. The
  required-empty-array grammar repair fixes a library integration defect; it
  does not relax the source schema or independent validator.

## Endpoints

Report fresh inference seconds, cache replay seconds, peak allocated GPU memory
where observed, accepted claims/questions, guard rejection counts, source-view
coverage, and downstream execution/compilation status. Preserve raw diagnostics
and exact source/model/configuration identities. Do not substitute lower
training loss or higher extraction counts for scientific correctness.

A model must pass fresh and cached offline execution on both positive smoke
documents with at least one accepted claim per document before it is considered
for a recommended extraction configuration. Failed candidates remain reported.
Qualified candidates can then be compared on the unchanged 30-case development
pack. Use one shared model session per batch and preserve every case outcome.

For an engineering recommendation, prefer the smaller candidate only if its
median fresh per-view latency is at most half the larger candidate's, it has no
more than two additional fully rejected cases out of 30, and all accepted source
spans pass the same verifier. Otherwise retain the larger candidate; if neither
qualifies, report that limitation rather than promoting one. These rules do not
establish semantic equivalence or expert-level validity, and no model is called
scientifically validated on this evidence.

Any model/configuration changes prompted by inspection of the 30 cases turn
those cases into explicit further development evidence. They cannot later be
described as an untouched final test.

## Transport repair and follow-up development checks

The first `source_schema` batch exposed incomplete quote output for a source
sentence containing embedded quotation marks. The independent schema validator
rejected that output. Preserve this batch as the original baseline, including
failed generations and rejected records.

The candidate `source_index` profile replaces generated quote strings with
integer selections from an immutable, document-bound sentence table. The
runtime restores the exact source string before applying the existing schema
and scientific source guard. This is a transport repair: it does not relax
relation, direction, negation, context, or future-work requirements. Synthetic
quoted, Unicode, newline, malformed-index, and cardinality cases are used for
implementation development; the biomedical prompt is not tuned to the 30-case
results.

After freezing the candidate implementation, repeat the two previously seen
qualification documents in new output directories. An initial CPU/float32,
four-thread check of the 1.5B model may run while the original GPU batch finishes.
It establishes CPU execution and source acceptance only. Do not compare its
latency to CUDA/bfloat16 measurements or apply the smaller-model speed rule to
mixed hardware. GPU qualification and any size comparison retain the original
hardware, output-token budget, seed, source guard, and two-document acceptance
rule.

If the candidate qualifies, run the same 30 inputs under the new profile and
audit source spans, input identities, artifact hashes, and workflow completion.
Call this a post-repair development regression, because a failure observed in
the original batch motivated the change. Report baseline and repaired-profile
outcomes separately, including fully rejected cases, retryable failures, and
terminal abstentions. More accepted records or valid serialization alone do
not establish better scientific interpretation.
