# Optional source-schema decoding candidate

The unrestricted Qwen failure retained in
`data/campaign_20261031/runtime-positive-v1` returned multiple claims, shortened
quotes, unsupported context, and a generated future question. The source guard
rejected them. This candidate constrains generation and leaves that guard intact.

`RuntimeConfig(decoding_mode="source_schema", max_new_tokens=384)` enables the
profile. Defaults remain `unconstrained` and 1,024 output tokens. The optional
dependency is `lm-format-enforcer==0.11.3`, using its documented JSON-schema
parser and Transformers prefix-token hook. See the
[upstream implementation](https://github.com/noamgat/lm-format-enforcer).

The dynamic schema permits zero or one claim. Quotes must be exact complete
source sentences, preserving prefixes and internal whitespace. Subject,
predicate, and object remain model outputs that must pass the independent
source guard. Context and comparator are null in this narrower profile.
Questions quote actual source sentences with explicit future/unresolvedness
cues; at most three are allowed. Empty output remains valid. Repeated and
incomplete sentences are not offered as choices, so coverage remains bounded.

Required field order and limited whitespace control output size. After decoding,
the payload is checked against the source schema and then against the unchanged
source/role/direction/negation guard. Malformed, truncated, or unsupported output
uses the existing bounded failure path. Missing or differently versioned
libraries fail clearly. This mode is extraction-only; requirement proposals
need an unconstrained runtime.

Cache/manifest identity includes mode, library version, dynamic schema hash,
profile instruction hash, helper implementation hash, and existing model/code/
config fingerprints. Tokenizer-prefix data are reused within a runtime; each
generation gets a fresh parser. `inference_fingerprint(document)` exposes this
same identity to qualification checks.

## Validation and limits

The targeted 96-test set passed with the isolated dependency target. Tests cover
schema bounds, prefix/context/question rejection, source whitespace, missing
dependencies, cache identity, and independent rejection of a format-valid wrong
relation. The pinned character parser was exercised. A tiny randomly initialized
CPU Transformers model also executed actual constrained `generate()` for an
empty-source schema. This validates integration, not pretrained semantic quality.
No GPU inference was executed by this implementation worker.

`scripts/v2/runtime_qualify.py` accepts `--decoding-mode` and
`--max-new-tokens`, retaining old defaults. It verifies fresh/replay model,
source, code, configuration, and schema identities. Failure retains a diagnostic
`FAIL` receipt and raw-generation cache. Optional `--require-claims` makes a
positive-source claim criterion explicit. Existing receipts are not overwritten.

The profile is not promoted by these tests. Fresh qualification on previously
used development documents is required after implementation freeze. Do not tune
on the 30-case evaluation pack or sealed holdout.

## Lead-run qualification

Run only after coordinating GPU ownership, from the integrated checkout, with a
new output path. Repeat for the second existing development document separately.

```powershell
$env:PYTHONPATH = (Get-Location).Path + ';C:\Codex\LitDataMatcher-v2\data\campaign_20261031\runtime-extras\structured-decoding'
& C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe scripts/v2/runtime_qualify.py `
  --model-dir C:\Codex\LitDataMatcher-v2\data\models\Qwen2.5-7B-Instruct\a09a35458c702b33eeacc393d103063234e8bc28 `
  --embedding-dir C:\Codex\LitDataMatcher-v2\data\models\all-MiniLM-L6-v2\1110a243fdf4706b3f48f1d95db1a4f5529b4d41 `
  --document C:\Codex\LitDataMatcher-v2\data\runtime-qualification\qualification_document.json `
  --output C:\Codex\LitDataMatcher-v2\data\campaign_20261031\source-schema-positive-1\QUALIFICATION.json `
  --device cuda --decoding-mode source_schema --max-new-tokens 384 --require-claims
```

Python socket/DNS access is denied during qualification; this is not an OS
firewall claim. A passing development smoke result does not establish
calibration, general extraction accuracy, or scientific answerability.
