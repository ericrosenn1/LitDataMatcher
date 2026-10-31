# Optional source-index decoder

`RuntimeConfig(decoding_mode="source_index", max_new_tokens=384)` selects an
experimental integer transport for the existing extraction guard. Defaults
remain `unconstrained`; `source_schema` keeps its existing quote transport.
Both constrained profiles are extraction-only. Native user questions continue
through the deterministic question compiler.

The current 30-case source-schema baseline remains unchanged. Its escaped-quote
failure motivated this general repair. Consequently, the 30-case pack is a
**post-repair development regression pack**, not an untouched test for this
candidate. Primary metrics, labels, inclusion criteria, and scientific guard
rules are unchanged. No case content was used to construct the synthetic tests
or to tune this prompt.

## Transport and provenance

The profile gives the model a numbered table of unique, complete source
sentences. The source selector is the same versioned selector used by
`source_schema`: repeated sentences and incomplete fragments are excluded,
with at most 128 choices. The prompt contains the indexed table instead of a
second copy of the full text. This omission is a coverage limitation, not
evidence that excluded text contains no relevant finding.

The model emits `sentence_index` instead of `quote`, plus the same subject,
predicate, object, direction, negation, status, context, and comparator fields.
Question indices must identify future-cue sentences and must be unique within
one output. Boolean, floating-point, string, null, negative, and out-of-range
indices fail validation. Duplicate JSON keys and nonfinite numbers also fail.
No invalid index is coerced, clamped, or replaced with another source sentence.

Deterministic lookup restores the exact public `quote` before the existing
source-schema and scientific guards run. Quoted group names, Unicode combining
characters, and embedded source newlines remain unchanged. An index does not
validate an inferred relationship: unsupported SPO, direction, or negation
still fails the independent guard.

Each runtime input is copied before fingerprinting. The mapping hash binds
sentence order, exact text, offsets, and future eligibility. Source identity is
checked again before mapping and publication. The cached `decoding_trace`
retains the table, raw output and parsed wire payload for each generation,
selected mappings, guard rejections, and any validated overflow. Offsets in
that table refer to the runtime input view; native analysis separately rebases
accepted evidence to parent-document coordinates.

The final per-view bound is one claim and three questions. Earlier accepted
IDs are retained and never overwritten across retries. Additional guard-valid
records remain in `validated_overflow`; they are not published as additional
accepted records. Failed generations do not erase earlier accepted partitions.
This bound is per runtime view, not a bound on an entire multi-view analysis.

Static backend identity includes profile, prompt, wire schema and mapping
versions, the new helper hash, the existing selector hash, and the pinned
`lm-format-enforcer==0.11.3` version. Dynamic identity additionally includes the
wire-schema and sentence-map hashes. The qualification gate compares static
identity while retaining the trial's document-bound identities. A new profile
requires a fresh qualification receipt.

## Validation and measured limits

The first synthetic round passed 24 tests. The expanded second round passed
137 tests, including the existing quote/unconstrained runtime and batch tests.
The final combined regression run passed 181 tests with zero failures or skips;
independent code review found no blocker. The scientific source-guard helpers
were checked for unchanged ASTs.
These are engineering checks, not scientific accuracy estimates. They cover
the escaped-quote failure, exact lookup, integer grammar and token prefixes,
wrong relation rejection, cache provenance, source mutation isolation,
cardinality across retries, and required-empty arrays.

A tiny randomly initialized CPU model executed actual constrained generation.
Explicit fixture-token preferences made its output deterministic while leaving
the grammar mask in force. This checks the generation integration; it is not
evidence of learned extraction quality. No pretrained weights or GPU inference
were run by the implementation worker.

`scripts/campaign_20261031/model_index_probe.py` uses seven synthetic examples.
All seven index payloads passed grammar and source guards; the original quote
grammar rejected the examples with internal quotes and embedded newlines.
Using the local Qwen tokenizer, serialized output fell from 454 to 357 tokens
across those examples (21.4%), while input templates rose from 3,518 to 3,676
tokens (4.5%). Combined token count increased slightly. These counts do not
demonstrate a latency improvement: prefill and autoregressive decoding have
different costs, and actual pretrained-model coverage and timing still require
coordinated measurement. Receipts are under
`data/campaign_20261031/source-index-validation-v1/`.

After the lead freezes and integrates the candidate, run qualification in a new
output directory using the existing positive development documents:

```powershell
python scripts/v2/runtime_qualify.py `
  --model-dir C:\path\to\pinned-Qwen `
  --embedding-dir C:\path\to\pinned-MiniLM `
  --document C:\path\to\existing-positive-development-document.json `
  --output C:\path\to\new-source-index-qualification\QUALIFICATION.json `
  --device cuda --decoding-mode source_index --max-new-tokens 384 --require-claims
```

Keep prior failures and the original 30-case run. A subsequent development
regression may measure the repair, but cannot retrospectively turn that pack
into an untouched evaluation.
