# Native serial analysis batches

The `batch` command runs the existing v2 pipeline for a frozen development case
pack. It shares one local extraction runtime and one retrieval encoder within a
single process. Case order is the input-file order; inference is serial. This is
execution infrastructure, not a new model, ranking policy, or scientific gold
standard.

```powershell
python -m litdatamatcher.v2 batch `
  --root C:\research\data `
  --inputs C:\research\cases.jsonl `
  --out C:\research\runs\batch-01 `
  --model C:\models\extractor `
  --embeddings C:\models\encoder `
  --device cuda --decoding-mode source_literal --max-new-tokens 384 `
  --runtime-qualification C:\research\qualification\QUALIFICATION.json
```

Choose a new output directory. Repeat the same command with `--resume` to reuse
verified terminal cases and resume interrupted stages. Model directories,
catalogs, code, runtime configuration, and exact input bytes must match. A new
configuration requires a new output directory. `--vector-cache PATH` optionally
uses the existing integrity-checked embedding cache. The session fits again if
the candidate records change between cases, without reconstructing its encoder.

Default runtime settings remain unchanged: unconstrained extraction and 1,024
maximum generated tokens. After qualification of the optional structured
decoder, `--decoding-mode source_schema --max-new-tokens 384` explicitly selects
that extraction profile. The command never switches decoding profiles silently.
The current development-qualified profile is
`--decoding-mode source_literal --max-new-tokens 384`; it uses the same source
schema and prompt with an escape-safe grammar. See the
[literal decoder guide](model_source_literal.md) and
[measured development outcomes](extraction_results.md). Its acceptance checks
do not establish scientific accuracy. CPU execution requires a matching CPU
qualification receipt and `--device cpu` on both commands.
User questions use the existing deterministic question compiler; no model-based
requirement proposal is invoked. `--requirements FILE` supplies the existing
explicit override for every case. `--fresh` requests fresh extraction on executed
cases; a verified completed case still skips on resume.

Backend qualification requires a v2.3 receipt with matching static model,
configuration, runtime, prompt, library, and decoder identities. Fresh and replay
identities/configurations must agree within that trial. The qualification
source's hash and dynamic sentence schema stay bound to the trial; a different
analysis document does not need the same text or sentence schema. Old
revision-only receipts remain historical evidence but do not qualify a changed
backend.
Use `--runtime-qualification PATH` to select a new immutable qualification
receipt without replacing the historical default at
`<root>/runtime-qualification/qualified_7b_pass1.json`. The selected file must
exist and its bytes are bound into native and batch resume identities.

## Input contract

Both the frozen `pipeline_inputs.jsonl` and `cases.jsonl` formats are accepted.
Each JSONL object needs a `document_id`, `id`, or `source_id` and nonempty `text`
or `abstract`. A `source_document` object can contain these fields. Optional
`question` or `query` becomes a caller-supplied question linked to that document;
this context link does not claim that the document answers it. All remaining
document/provenance fields are retained. An explicit `case_id` is recommended;
otherwise a deterministic record hash supplies it.

Inputs are limited to 1,000 development cases and 50 MiB. Declared non-development
splits, duplicate IDs including case-only variants, Windows reserved names,
unsafe paths, duplicate JSON keys, and nonfinite JSON numbers are rejected.
Unicode line/paragraph separators inside scientific text are preserved as text.
This command does not open or sample sealed/evaluation-candidate corpora.

## Output and recovery

- `BATCH_STATE.json` binds inputs, resolved runtime settings, shared model/code
  identities, case status, per-attempt results, and output SHA-256 receipts.
- `BATCH_RESULT.json` reports pass, partial, failed, retryable, and verified-skip
  counts after a complete pass through the pack.
- `inputs/<case_id>.jsonl` preserves each materialized document.
- `cases/<case_id>/attempt-0001/` contains native reports, source spans, candidate
  outcomes, dossiers, and stage checkpoints. Resume reuses this attempt's valid
  stages; it does not rerun successful extraction unnecessarily.
- `failures/<case_id>-<attempt>.json` retains technical failures or partial
  results, including timestamps and error details. Later success does not delete
  earlier failure receipts.

Technical case failures normally allow later cases to run. `--stop-on-error`
stops on an exception or retryable native result. Keyboard interruption always
stops promptly and persists state. Exit status is nonzero if any case failed or
remains retryable. A terminal `PARTIAL` result from source-guard abstention is
retained explicitly; it is not scientific acceptance or a fabricated negative.

Terminal case reuse checks native output bytes and checkpoint receipts. Native
document ingestion is now a resumable stage: changed/malformed derived ingestion
artifacts are quarantined before rebuilding from the unchanged original input.
One session belongs to its creating process/thread, prohibits nested claims, and
checks configuration/source identity at case boundaries and full content hashes
at batch completion. It is not a cross-process model server.

## Source evidence policy

Normalization policy `structured-source-field-evidence-v1` retains raw adapter
fields but promotes organism/assay capabilities only from explicit source-backed
capabilities or `field_evidence` with value, observation status, mapping, and
field locator. A source URL by itself is insufficient. Explicit absence and
already supplied non-unknown capability states are not overwritten.

The qualified GEO ESummary adapter may supply exact `taxon` and `gdstype` fields,
with accession/UID and adapter provenance checks. Mixed-organism text stays
mixed. “Expression profiling by high throughput sequencing” does not become
bulk or single-cell RNA-seq. PRIDE/Metabolomics Workbench records retain actual
structured source fields and API pointers; repository-level assay defaults do
not become observed measurements. Registry study design remains a registered
design, and does not establish observed outcomes, independent donors, or power.

This repair tightens evidence eligibility. It does not change ranking weights,
scientific thresholds, source-guard rules, or the meaning of unknown measurements.

## Validation boundary

Focused tests use explicit synthetic source records and runtime/encoder stubs to
check reuse, ownership, changed-input rejection, failure continuation, ingestion
repair, Unicode preservation, and source-evidence fidelity. They do not measure
real-model accuracy. A real 30-case GPU run is a separate lead-owned step after
runtime qualification; this implementation task does not execute it.
