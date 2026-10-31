# Scientific evaluation protocol draft and acceptance-gap audit

**Review-ready draft; final scientific protocol approval and scientific acceptance
remain pending.** Evidence cutoff: implementation
`b9f1cf3905e693953a64a74888f400504f87d2fc`, reviewed October 31, 2026.
The [audit receipt](../validation/scientific-evaluation-readiness-20261031.json)
binds inputs and independent checks. This downstream assessment of implemented
functionality runs no evaluation, changes no criteria and accesses no reserved
final-evaluation data.

## Existing protocol authority

The exact master request's sections 7, 8, 11 and 12 require real compatibility
demonstrations, relevant baselines, preserved final evaluation and independent
scientific evidence. [EP-20260907-1](../v2/EVALUATION_PROTOCOL.md) and the
[build-specification gates](../v2/build_spec/08_VALIDATION_AND_ACCEPTANCE_GATES.md)
already provide an evaluation structure. This draft extends their review
coverage, not their scientific definitions. Their historical scope/reservations
are not an approved final protocol for the expanded application.
[E04](../v2/EVALUATION_ROUND2.md) records exposure of an earlier reserved family;
it must not be represented as untouched. The current master request reserves
GSE279879. No reservation is opened or replaced here.

Preserve the existing pilot targets and their original scope:

| Existing EP requirement | Unchanged interpretation |
|---|---|
| Zero forbidden direct-fit promotions and source/identity/lineage/consequential-negation/unsupported-novelty errors | A consequential error blocks the declared path; speed cannot compensate. |
| At least 95% qualifier/capability correctness over at least 40 independently enumerated fields and 10 source families | Below-floor checks are exploratory; these are engineering quality targets, not biological power. |
| Recall@10 at least 90% over at least 20 fully judged primary queries with usable candidates; 10 transfer queries in the expanded protocol | Retain no-fit queries separately. Missing judgments are not negative labels. |
| Zero compatibility-aware invalid-top matches; descriptive precision@5 and nDCG@5 | Use identical judged candidate universes and preserve existing graded fit definitions and unknowns. |
| At least six distinct dossiers, two substantive evaluations per central worker, three integrated rounds, and two consecutive rounds without material gain before final evaluation | Keep the material-gain rule: consequential defect correction, 2 percentage points in a known-denominator primary metric, or 10% lower median latency without quality regression. |

The [Phase 2 metadata](../v2/final_audit/PHASE2_BENCHMARK_PROTOCOL.md),
[October retrieval](retrieval_protocol.md) and [extraction](extraction_protocol.md)
protocols remain narrower development experiments. Their engineering bounds and
promotion rules do not replace scientific acceptance or authorize holdout use.

## Prospective execution draft

Retain the existing process: freeze the approved scientific scope, complete
question/candidate universe, publication/cohort-family split, dated source
coverage, software/model/configuration identities, label protocol and denominator
rules before final evaluation. Resolve versions, reused cohorts and repository
mirrors before claiming independence. Preserve exclusions, failed acquisitions,
empty results, contradictions and unknowns. Do not tune on final cases.

Evaluate existing endpoints separately: extraction support/qualifier correctness;
recall only with independently enumerated source propositions; capability and
comparator correctness; candidate recall/ranking, invalid-top and no-fit/partial
behavior; lineage/traceability; runtime/resources. Independent opportunity review
must distinguish question validity, whether literature addresses it, and whether
data meet its requirements. No new composite endpoint, accuracy threshold,
biological estimand or sample-size calculation is adopted here.

Compare existing BM25, pretrained dense/hybrid and requirement-aware methods on
identical frozen inputs and candidate universes. Compare rules/model modes on
matched source views and record configuration differences. Preserve abstention,
failure and null/no-gain results separately. Report numerators, denominators,
family counts and appropriate uncertainty. Scores and textual groups remain
review aids, not probabilities or independent biological evidence. Calibration
requires genuine labels and adjudication; neither is supplied by this draft.

## Independent acceptance-gap audit

| Executed evidence | Defensible conclusion and remaining gap |
|---|---|
| [Twelve descriptive comparisons](descriptive_comparisons.md), including twelve live API results independently checked for essential joint-observation/ascertainment MATCH | The initial positive-development milestone is demonstrated under full explicit comparative contracts. The cases reuse two UCI datasets; they are not twelve independent studies or newly unanswered opportunities. All retain statistical adequacy UNKNOWN. |
| Original twelve processed-table association cases and 30-case/140-candidate seed work | Original association full-fit count remains **0/12**. The preserved model campaign's 140 candidate rows remain unqualified. Unsupported timing and other missing requirements were retained; descriptive examples do not silently reinterpret the original questions. |
| [Qwen development receipts](../../benchmarks/campaign_20261031/model_qualification.json): 30 cases, 11 PASS, 19 terminal PARTIAL, 20 source-validated claims, 30 verified resume skips | Runtime/source-guard evidence, not expert accuracy. The 1.5B configuration failed both positive qualification documents; no full-batch or model-size speed advantage follows. Historical decoder timing differences are not controlled causal speed estimates. |
| Rules discovery on the 30 source documents retained 20 passages, all automatically underdetermined; saved live model job `041f39b700814d92aff0a51867dea689` processed two documents, retained **one** question and zero candidates, with COMPLETE/QUESTION_UNDERDETERMINED status | Source-linked extraction and safe abstention are exercised. These distinct observations are not a controlled rules/model accuracy comparison. Scientific question validity and downstream usefulness remain independently unjudged. |
| [Metadata-only run](../validation/question-discovery-scale-20261031.json) retained 744 missing abstracts as not extracted; [10,000-input run](../validation/question-discovery-10000-scale-20261031.json) retained 935 passages: 856 SUCCESS, 9,120 EMPTY, 24 lifecycle exclusions | Source text absence, empty extraction and excluded lifecycle records remain distinct. One prior batch was reused and nine newly extracted. [Independent reconciliation](../validation/question-discovery-10000-audit-20261031.json) verifies counts, null retention and hashes. Topic-selected mixed partial snapshots establish engineering scale, not accuracy or absence of research gaps. |
| [Global grouping](../validation/question-discovery-10000-global-20261031.json): 935 retained passages form 931 review groups at unchanged 0.9 threshold, without re-extraction | Proposed textual grouping only; no scientific equivalence, independent support or global novelty claim. |
| [Dataset ranking](../../benchmarks/campaign_20261031/dataset_ranking.json): 12 availability queries, 121 candidates; MRR@5 BM25 0.5000, dense 0.7708, hybrid 0.5417, requirement-aware 1.0000 | Deterministic availability labels test known-target/contract consistency, not expert suitability. Hybrid underperformed dense. [Source-paper retrieval](../../benchmarks/campaign_20261031/retrieval_development.json) retains BM25 as best overall despite encoder improvement over its parent. Neither experiment proves end-to-end opportunity quality. |
| Immutable corpus chunk receipt: 957/1,621 shards and 28,710,000 current records within that applied prefix; integrated API/browser examples | Large searchable partial coverage exists. Historical 45,488,930 citation events are not current unique papers. The reviewed receipt requires separate full integrity validation; full current-view/FTS acceptance is not established by it. Semantic retrieval reranks a bounded lexical shortlist, not the whole corpus densely. |

The [original frozen review packet](scientific_review_packet.md) contains 42
cases, 47 question rows and 152 candidate-review rows. The separate
[12-case addendum](comparative_review_addendum.md) contains 180 blank reviewer
cells. Together they provide review coverage, not 54 independent reference
studies. Neither packet contains invented expert labels; development cases
cannot be reclassified as an untouched final set.

## Decisions requiring scientific ownership

| Decision to record before final evaluation | Current state |
|---|---|
| Confirm primary/transfer scientific scope and reconcile historical/current family reservations without exposing final data | Pending; no replacement cohort, final case list or sampling design selected here. |
| Specify the complete judged candidate universe, case/family denominators, dependence handling and treatment of partial/unknown labels under existing endpoints | Pending; two reused UCI datasets cannot supply independent denominators. |
| Assign genuine domain reviewers, qualifications, review/adjudication policy and disagreement procedure | Pending; forms remain blank. Independent code review is not human scientific annotation. |
| Define dated literature coverage and evidence for answered, partial, contradictory and insufficient-coverage judgments | Pending; no global novelty conclusion is justified. |
| Determine study-specific unit independence, confounding, estimand and precision/power requirements where the intended analysis needs them | Pending; row counts and this draft supply no such scientific assumptions. |
| Confirm existing thresholds' applicable scope, prospective comparison/uncertainty plan and protocol approval record before sealed evaluation | Pending; substantive revision requires justification/versioning before new evaluation, retaining earlier criteria and results. |

**Acceptance conclusion:** development feasibility and engineering evidence are
substantial; independent scientific reference judgments, final protocol approval,
calibration and untouched final evaluation are **not completed**. This review
does not certify an exact-head whole-product release: corpus completion and
remaining application/refinement checks require their own current receipts.
Pending scientific judgments do not invalidate the narrowly stated twelve-case
development demonstration, and cannot be replaced by fabricated positives or
model-generated expert labels.
