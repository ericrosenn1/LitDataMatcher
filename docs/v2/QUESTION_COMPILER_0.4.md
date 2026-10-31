# LitDataMatcher v0.4 question-to-data-requirements compiler

`litdatamatcher-v2 compile --question "Does semaglutide increase risk of NAION in adults?" --out compiled.json`

The native compiler produces a versioned answer specification, analysis-plan
requirements, role mappings, exact supporting spans, provenance categories and
diagnostics. It does not predict whether the risk increases. When `analyze`
has no expert requirements, it now compiles every extracted or supplied
question before candidate retrieval; `question_compilations.jsonl` preserves
the full contract, while `questions.jsonl` contains the validated assessor
projection. An empty contract produces `NO_ASSESSABLE_CONTRACT`, not a best
scientific match.

Expert requirements remain optional. They are retained as `USER_SPECIFIED`;
when they conflict with an automatically compiled field, the automatic field
is retained as non-effective provenance and the conflict is recorded.

The current local rules cover comparative, associational, longitudinal,
prediction, perturbational/mechanistic, entity-discovery and literature-only
questions. They require only the capabilities necessary to answer the stated
question; numeric power, effect-size and follow-up assumptions remain explicit
unresolved parameters unless supplied from valid inputs.
