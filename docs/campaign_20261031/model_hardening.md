# Source-grounding and compiler hardening

This is a bounded engineering and development-data comparison against research
incumbent `e592407`. It does not establish expert-calibrated extraction accuracy,
clinical validity, or a stronger trained model. The pretrained models and all
ranking weights are unchanged.

## Changes

- The extraction guard locates unique, token-aligned subject, predicate, and
  object spans in order. Roles crossing a clause boundary remain rejected.
- Direction and negation refer to the selected relation's clause. `no_change`
  requires support in the selected predicate, not another clause's finding.
- A repair-attempt generation error preserves accepted source records. The
  manifest retains its error class/count; initial inference still fails when
  no valid partition exists. Cache replay preserves that history.
- Risk-reference resolution requires a unique explicit `risk of/for TARGET`
  or `TARGET risk` mention. Question and antecedent context spans are retained.
  Competing or absent targets produce an underdetermined contract. Explicit
  user outcomes retain authority.
- Bounded predictive, associative, and perturbational inflections are recognized.
  Copular parsing is anchored to an interrogative clause; it cannot promote an
  adverb after a declarative `is` to an exposure.
- Population phrases are separated from generic outcomes and comparators.
  Multiple species require disambiguation. `Child-Pugh` is not a pediatric
  population. Nested phrases such as `changes in mood` remain intact.
- Original question whitespace and source offsets are preserved; normalization
  is a separate view.
- A task word alone cannot make an automatic contract complete when named
  target roles are missing. Partial roles remain visible with an explicit
  `UNRESOLVED_TARGET_ROLES` diagnostic, and matching abstains. Explicit user
  contracts retain their existing authority. Identical user/automatic fields
  do not count twice; duplicate user fields fail clearly.

The prompt is `source-extractive-v2.4`; extraction schema is
`semantic-extraction-v2.2`. Compiler output is
`question-requirements-compiler-1.1` with `source-bound-roles-1.1`.
`question-method-rules-1.0` values and ranking weights are unchanged. New code
and prompt fingerprints invalidate prior inference cache keys and runtime
qualifications. Fresh bounded development qualification is required before
claiming the changed prompt's pretrained-inference performance.

## Executed comparisons

Final receipt:
`C:\Codex\LitDataMatcher-v2\data\campaign_20261031\model\comparison_rounds_1_2_v4\MODEL_COMPARISON.json`.
It records source/input/output hashes, software versions, and case differences.

| Check | Incumbent | Candidate | Interpretation |
| --- | ---: | ---: | --- |
| Engineering challenges passed | 10 / 27 | 27 / 27 | Synthetic contract checks, including simulated retry failure |
| Development records | 2,143 | 2,143 | Previously selected allocation, no new selection |
| Source-span mismatches | 0 | 0 | String/offset consistency, not semantic correctness |
| Structured PICO strings retained | 1,662 / 1,662 | 1,662 / 1,662 | Literal question-field consistency |
| `COMPLETE` | 1,780 | 1,708 | Status count, not accuracy or recall |
| `QUESTION_UNDERDETERMINED` | 363 | 435 | Explicit abstention is a valid outcome |

The development input contains 1,662 Evidence Inference 2.0 records, 135
EvidenceBench records, and 346 PubMedQA records. Its SHA-256 is
`3c09a30881e193f6229f539c01685dbb5dd7cae77b31665488537c8f061286e6`.
Original corpus partition labels embedded in IDs were not used to reselect data.

Seven previously underdetermined questions gained explicit target roles.
Seventy-nine previously complete records now abstain because target roles were
not parsed: 60 lacked both roles, 13 lacked intervention/exposure, and six lacked
outcome. This is a transparent coverage tradeoff, not proof of corrected
scientific errors or changed true recall. A total of 143 records changed roles,
purpose, diagnostics, or status. Candidate processing plus artifact writing took
1.17 seconds locally; no comparative performance claim is made.

The 1,662 PICO checks inspect literal fields in supplied question text. They are
different from the prior 1,601 questions with separately available Evidence
Inference relation annotations. Neither is an expert-validated full requirement
benchmark.

Manual inspection rejected an earlier candidate that treated `not consistently`
as an exposure and `Child-Pugh` as a population cue. Its receipt remains under
`comparison_rounds_1_2_v1`. Repaired comparisons remain as v2 and v3. Independent
review caught a nested-preposition truncation, repaired and evaluated in v4.
No unfavorable attempt was overwritten.

Validation: 74 targeted tests passed across `test_semantic_runtime.py`,
`test_question_compiler.py`, `test_campaign_model_hardening.py`, and
`test_v2_integration_contracts.py`; targeted Ruff checks passed. The original
38 tests in the first two modules also passed before changes. These tests did
not substitute for the separate development-data comparison.

## Limits and next qualification

The guard deliberately abstains on repeated roles, passive order, and unresolved
clauses. The compiler handles a bounded vocabulary and coarse species roles;
age-stratum eligibility remains unvalidated. Terms such as stimulation can be
metaphorical outside experimental biology. Recognizing a task family is not
proof of appropriate study design. Accepted interpretations still need review.

No model weights were trained/downloaded; no GPU inference, corpus normalization,
sealed holdout access, `evaluation_candidate` inspection, or external service
call occurred. The larger corpus requires its independent integrity gate before
retrieval/learning. The next model step is paired fresh inference on approved
development sources with new fingerprints, reporting unsupported relations,
abstention, source coverage, and latency separately.

## Reproduction

Run from the changed checkout with a new output directory. The helper refuses
to overwrite receipts and requires an adjacent development-only baseline receipt.

```powershell
$env:PYTHONPATH = (Get-Location).Path
& C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe scripts/campaign_20261031/model_evaluate.py `
  --baseline-ref e592407 `
  --development-records C:\Codex\LitDataMatcher-v2\data\v0.4_compiler_20260913\baseline_current_compiler_dev_v1_20260914_r2\deterministic_compiler.jsonl `
  --output C:\Codex\LitDataMatcher-v2\data\campaign_20261031\model\new-comparison
```
