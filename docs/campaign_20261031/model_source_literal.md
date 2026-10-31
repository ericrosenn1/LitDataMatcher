# Source-literal decoding

`RuntimeConfig(decoding_mode="source_literal")` is an opt-in transport profile.
The native `analyze`, `batch`, and `scripts/v2/runtime_qualify.py` commands
accept `--decoding-mode source_literal`; `--max-new-tokens 384` is an explicit
qualification setting, not a changed default. The optional dependency remains
`lm-format-enforcer==0.11.3`. No model files are downloaded by this profile.

The profile preserves the exact original `SYSTEM_PROMPT` plus
`PROFILE_INSTRUCTION`, public `source_schema`, source-sentence selection,
field order, per-attempt array limits, null context/comparator, and independent
relation/direction/negation guard. It replaces the JSON enum transport with
public LMFE `StringParser`, `SequenceParser`, and `UnionParser` composition.
Each quote is an entire `json.dumps` literal; backslashes, quotes and controls
therefore remain escaped. Unicode quotes have both literal and equivalent ASCII
escape spellings. The latter provides a reachable path when a byte-level
tokenizer splits a Unicode character. Parsing restores the original source
string without a repair or substitution.

Generic subject/predicate/object strings still use LMFE's bounded JSON string
parser. The composed grammar forwards the standard active-string acceleration
only when its branch is unambiguous. A local prefix wrapper excludes raw C0
control tokens inside JSON strings while allowing legitimate external JSON
whitespace and quote transitions. It does not mutate LMFE, the tokenizer, or
shared tokenizer data. Arbitrary printable Unicode in generic strings remains
subject to strict JSON/schema validation and the unchanged source guard.

Final output retains the first guard-accepted **one claim and three questions
per source view** across attempts. Earlier IDs are never overwritten. All raw
attempt payloads, guard rejections, and validated overflow remain in
`decoding_trace`; cache artifacts and raw diagnostic files preserve provenance.
The private source snapshot is hashed and checked before/after processing.
Duplicate JSON keys are rejected. Source schema, implementation, library and
configuration hashes distinguish this profile from both earlier modes.

The existing `source_schema`, `source_index`, and unconstrained modes remain
available. Earlier failed and partial runs remain historical evidence. The
7B literal configuration passed both positive development documents and the
30-case run completed with no retryable technical failures. It is the current
development-qualified transport; this does not establish scientific accuracy
or eliminate source-guard abstentions. See the
[measured outcomes and limits](extraction_results.md).

## Bounded evidence

Evidence directory:
`C:\Codex\LitDataMatcher-v2\data\campaign_20261031\source-literal-validation-v1`.

- Initial profile regression round: 102 passing tests, no skips.
- Expanded profile/runtime round: 138 passing tests, no skips, including a real
  tiny random CPU model with explicit fixture-token preferences. That test
  validates Transformers integration, not pretrained model ability.
- Qwen tokenizer-only probe: 11/11 synthetic fixtures have a complete reachable
  exact serialization with this profile; the old source-schema parser reaches
  2/11. Cases include quotes, backslashes, Unicode/combining marks, newline,
  CRLF, tab, NUL, emoji, and 8/128 common-prefix sentence choices.
- Some literal Unicode encodings remain unreachable through LMFE's byte-token
  bridge; their ASCII-escaped equivalents pass and decode to identical strings.
- Successful final synthetic token walks take roughly 0.19–0.26 seconds after
  shared tokenizer initialization; these are short CPU prefix traversals, not
  inference speed, throughput, or scientific-quality measurements. Early
  failures in the old profile are shorter and are not fair latency comparisons.

`python scripts/campaign_20261031/model_source_literal_probe.py --tokenizer <local-model-directory> --output
<new-receipt.json>` reproduces the synthetic tokenizer experiment without model
weights. Small fixtures and 128 choices do not establish production peak memory
or general latency guarantees. The extraction evidence report separately records
real-model measurements.

No biological prompt tuning or 30-case data was used to build this grammar.
The previously inspected 30-case pack is development/regression evidence after
repair, not an untouched final test. Scientific criteria and primary metrics
are unchanged; genuine expert validation remains a separate requirement for
claims about scientific accuracy.
