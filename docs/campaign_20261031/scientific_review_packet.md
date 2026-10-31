# Development scientific review packet

The generated packet reuses the 30 preserved source-linked development cases
and 12 processed-table availability examples. It contains 42 cases, 47 exact
questions (35 native variants plus 12 availability questions), and 152 candidate
review rows (140 preserved native rows plus 12 selected processed-table pairs).
Observed evidence covers 74 distinct dataset accessions. All 185 source files
read by the generator are hashed; native artifacts also match their existing
batch receipts. No model, scientific analysis or sealed evaluation is rerun.

`question_review.csv` and `candidate_review.csv` leave genuine reviewer identity,
full feasibility, unresolvedness, missing requirements and justification blank.
Source IDs, article text/locators, exact questions and complete historical
compiled requirements are retained in `review_cases.jsonl`. Dataset observations
are deduplicated in `dataset_evidence.jsonl`. Historical machine decisions and
question-status interpretations are separated into `machine_decisions.jsonl`.
The protocol explains the limits of this separation; it does not claim formal
blinding while machine-generated requirements themselves are under review.

This packet is technically validated and scientifically unreviewed. The
availability examples reuse only two UCI tables and do not establish novel
unanswered questions. Existing scientific criteria remain unchanged. GSE279879
and every other sealed final-evaluation asset remain outside the packet.

The human-dependent scientific-validation steps are to assign genuine domain
reviewers, obtain justified labels and adjudicate disagreements, define treatment
of uncertain/dependent cases, and approve the prospective evaluation protocol
before accessing an untouched final set. This development packet cannot be
reclassified as that final set. No expert identities or labels were invented.

```powershell
& 'C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe' -m scripts.campaign_20261031.prepare_scientific_review --cases 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\research-cases-v1' --batch 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\native-batch-literal-v1' --availability 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\dataset-qualification-v2' --out 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\scientific-review-new'
```

Use a fresh output directory; the generator refuses to overwrite prior reviews.
The executed packet is at
`C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\scientific-review-v3\LitDataMatcher-development-scientific-review.zip`.
It contains the forms, evidence, protocol, vocabularies, source/output manifests,
technical validation, structured task state and generator. ZIP member bytes and
checksums were verified. Archive SHA-256:
`915372311bc45337fea7f8daa25c78981253c4619bf71a797c7605bcadccfc9b`.
The committed receipt is `benchmarks/campaign_20261031/scientific_review_packet.json`.
