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

Validation: 26 focused discovery/compiler tests passed. A fresh execution over
the 30 existing development source documents retained 20 passages: 14 research
questions, five source-stated unresolved passages and one limitation, without
technical failures. All 20 remained underdetermined under the unchanged
scientific compiler. These are development execution observations, not expert
correctness labels or evidence that 20 new opportunities were discovered.
The output is retained under
`C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\questions`.
