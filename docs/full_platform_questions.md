# Source-grounded question discovery

`litdatamatcher.question_discovery` provides the application-facing discovery
and compilation functions without replacing the qualified extraction runtime
or changing the established scientific requirement rules.

```python
from litdatamatcher.question_discovery import (
    discover_questions, compile_research_question, check_addressed_evidence,
)

discovery = discover_questions(documents, max_documents=100, max_questions=200)
# To use the existing qualified local Qwen runtime explicitly:
# discovery = discover_questions(documents, runtime=runtime, cache_root=cache)
contract = compile_research_question(
    "Does insulin increase glucose uptake in humans?",
    user_constraints=[{"field": "paired", "expected": True, "essential": False}],
)
```

Documents require a stable `document_id`, `id`, or `source_id`, source text or
abstract, and optionally a source locator, URL, citation and provenance. Exact
source offsets, UTF-8 source-text SHA-256, source identities and model manifests
are retained. Deterministic extraction is explicitly identified as such. A
model failure never silently changes to heuristic extraction. The existing
source guard revalidates model quotations, while repeated ambiguous sentence
strings are omitted by the established source selector.

Before extraction or addressing checks, the existing literature-integrity
consolidator derives lifecycle/version eligibility. Retraction, correction,
version-review and explicit ineligibility decisions exclude the source before
model invocation. Supplied lifecycle snapshots, including merged-source member
snapshots, remain preserved; missing relation payloads cannot erase a prior
explicit exclusion. Excluded sources have `EXCLUDED_SOURCE_LIFECYCLE` document
status and a separate coverage count. Retained questions carry current integrity,
source provenance and supplied lifecycle snapshots, without promoting
`REQUIRES_SOURCE_REVIEW` into established evidence eligibility.

Source-stated unresolved passages, limitations, and interrogative research
questions have different categories. A question mark does not establish that
a question is unanswered. Limitations are not rewritten into invented
hypotheses. Normalization, exact/near-duplicate grouping, and lexical checks for
related result passages retain their review status. They never establish
global novelty, independent evidence, or a definitive answer. Original member
questions remain available even when grouped.

`compile_research_question` retains the existing compiler's requirements,
provenance, ambiguity handling and essential constraints. Its
`matching_requirements` view also includes explicit soft preferences. Proposed
alternative designs are labeled `PROPOSED_UNASSESSED`, with reviewable
assumptions; no alternative silently changes the active contract.

Requests are bounded to 1,000 documents and 1,000 retained questions. Individual
source views use the existing 128-sentence source selector and 200,000-character
input ceiling. Large articles need explicit upstream windows. Failed views,
empty results and truncated result sets remain distinguishable.

Metadata-only papers without source text or an abstract are recorded as
`METADATA_ONLY_NOT_EXTRACTED`; titles are never substituted as paper evidence.
Malformed records and unresolved duplicate identities have per-input failures,
so valid peers continue. Addressing checks separately report missing-text
coverage without turning it into a negative scientific finding or a transient
model failure.

Validation: 27 focused discovery/compiler tests passed. A fresh execution over
the 30 existing development source documents retained 20 passages: 14 research
questions, five source-stated unresolved passages and one limitation, without
technical failures. All 20 remained underdetermined under the unchanged
scientific compiler. These are development execution observations, not expert
correctness labels or evidence that 20 new opportunities were discovered.
The output is retained under
`C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\questions`.

The final browser suite passed 41 checks across desktop and mobile, with one
intentional duplicate mobile computation skipped. It reopened a preserved real
job and downloaded its dossier; no new computational job was needed. Results now
display processing completeness and extraction coverage independently of queue
success, including metadata-only, lifecycle-excluded and failed documents.

A bounded scale demonstration replayed 1,000 distinct PubMed breast-oncology
records from the same committed partial corpus snapshot (563 applied shards).
The source audit found an upstream index title fallback: 744 records had no
abstract. That derived source-view issue was corrected explicitly, preserving
every other native field and the original diagnostic run. The corrected run
processed 256 available abstracts, recorded 744 metadata-only documents, and
retained 18 literal passages from 17 records; 239 processed abstracts contained
no configured question/limitation cue. There were no extraction failures or
truncation. All 18 span, hash, citation, locator and provenance checks passed.

Extraction plus grouping took 0.082 seconds in this single small-text run;
sampled process RSS peaked at 48.17 MiB (5.92 MiB above its extraction baseline).
These measurements do not establish large-corpus throughput, reviewed extraction
accuracy, global novelty or scientific validity. Missing abstracts were not
counted as extracted documents or negative findings. The first title-fallback
run is retained as an explicitly superseded diagnostic and must not be cited as
abstract-extraction evidence.

The reproducible no-retrieval script is
`scripts/validate_question_discovery_scale.py`; the corrected measured receipt
is `docs/validation/question-discovery-scale-20261031.json`. The source snapshot,
correction ledger, questions, validation, manifest and verified handoff archive
are retained under
`C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\question-scale-1000-v2`
and its adjacent `.zip`. No model was loaded or expensive successful run repeated.
