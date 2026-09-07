# Validation, acceptance and refinement

## Reporting contract

Build a validator that reads actual run/test artifacts, checks their schemas and provenance, and emits gate IDs, PASS/FAIL/NOT_RUN, source commit/config/model/input fingerprints, executed evidence, unresolved issues, and readiness. It must not infer success from file presence, README claims, a worker's prose, or a nonzero count alone. Product-required NOT_RUN is not PASS. Infrastructure availability and scientific calibration are reported separately.

`templates/BUILD_SPEC.json` fixes numeric coverage floors. `templates/ACCEPTANCE_REPORT.schema.json` defines the reporting envelope. The product validator is to be implemented by Codex; the package integrity verifier is not that validator.

## Required product gates

| Gate | Required executed evidence |
|---|---|
| G01 Installation and user inputs | Clean isolated install from the delivered distribution, outside the source checkout; a new local document, topic and explicit question can be used through documented interfaces |
| G02 Real literature | Coverage floor achieved; valid identifiers, accessible source snapshots, full-text parse validation and duplicate/version accounting |
| G03 Real dataset catalog | Accession-level independent study floor achieved; at least GEO and one sequencing-repository path exercised without counting mirrors as new studies |
| G04 Capabilities and data files | Sample/group profile floor; actual inspected processed files from distinct studies; feature/sample alignment, unit counts and usable contrasts validated |
| G05 Real semantic runtime | Fresh application-process model extraction on previously unprocessed input; runtime/model revision recorded; not prewritten agent output or regex-only substitution |
| G06 Evidence and normalization | Claim schema, negation/direction/context and quote support checked; entity ambiguity represented; source offsets/locators and provenance survive persistence |
| G07 Questions and dated gaps | Automatic explicit and cross-document gap generation plus user-question mode exercised; answered, partial, contradictory and insufficient-coverage cases handled |
| G08 Experimental matching | Essential requirements honored; missing metadata distinct from incompatibility; no-fit and partial-fit responses correct; multi-dataset joint-observation constraints tested |
| G09 Evidence compiler | At least one imported external structured resource queried with literature; dependence, contradiction, indirect evidence and integration-mode tests pass; real valid numerical combine/harmonization demonstration plus invalid-combination abstention |
| G10 Evaluation and ranking | Baseline/hybrid/compatibility-aware comparisons, label provenance, study-grouped holdout, hard-negative and source-disjoint tests, full opportunity review, score explanations and nonprobabilistic heuristic labels |
| G11 Incremental and offline | Source update invalidation and idempotence; warm offline replay with network blocked; separate fresh local inference while offline; no hidden downloads |
| G12 Failure recovery | At least two interrupted stages resume without duplicate scientific records or lost good artifacts; HTTP 429/transient errors/schema drift/corruption/inference failures handled correctly |
| G13 Resource and concurrency | Useful jobs demonstrably overlap; no shared-writer corruption; resource-pressure/backoff and task-owned cleanup tested; numeric exits and bounded logs captured |
| G14 Scientific and engineering review | Actual independent Codex reviews with reproduced findings; no unresolved critical/high issue in declared functionality; minimum substantive refinement passes completed |
| G15 User-facing outputs and security | Report generated from stored real outputs with traceable claims and escape-safe text; secret/license/prompt-injection checks; commands and false-success detection tested |
| G16 Delivery and readiness | Clean delivery archive/wheel/source integrity, versions/locks/manifests/notices, final real run and machine readiness report agree; functional-versus-hardened state and stopping reason correct |

For a numeric integration example, choose a valid bounded task the data support: e.g., comparable effect estimates or a justified feature/sample harmonization. Do not claim arbitrary unpaired omics matrices can be merged. A designed synthetic case can test rejection logic, but the positive demonstration must use real data with a documented analysis contract.

## Development operations, reported separately

- O01: original source/work preserved, baseline recorded, worker isolation and lead-only integration tested.
- O02: safe checkpoint commit and remote ref match demonstrated; no main change, secrets or bulk data in staged/pushed content. Temporary offline push backlog is visible.
- O03: valid live state, exact continuation command, owner lease and pause/capacity-wait handling demonstrated.
- O04: scheduled supervisor local access, healthy no-op, deliberate pause, abandoned-job repair, stale ownership, takeover conflict prevention and real resume tested. Before installation it is `PREPARED_NOT_ENABLED`; unavailable platform control is `UNAVAILABLE`, never PASS.
- O05: current delivery and ongoing deterministic jobs have explicit owners and stop conditions; supervisor disabled/idle at completion where supported, otherwise exact remaining UI action reported.

Missing scheduling UI must not block development of the product or create a report-only replacement. Do not describe unattended recovery as verified when O04 is not PASS. Product readiness may coexist with a visibly unconfigured schedule; full unattended-development readiness may not.

## Test layers

**Unit/contract:** schemas and finite numbers; IDs; raw/normalized offsets; type and null behavior; variable/mapping serialization; score components; deterministic functions; hash and cache invalidation; round trips preserving nested provenance; supported legacy migrations. Do not silence old regression failures without deciding whether behavior is wrong or compatibility intentionally changed.

**Live/replay:** each selected adapter must execute a live query and replay captured permitted responses. Test paging, date updates, partial metadata, incomplete downloads, archived/retracted/corrected records when available, and optional-source failure. Network-free mocks alone do not establish current source compatibility.

**Metamorphic:** input reordering should not change identity; duplicate evidence should not increase independent support; adding metadata can resolve unknowns; adding contradictory data can alter conclusions; changing negation/dose/tissue/assay must affect interpretation appropriately; model/config updates must invalidate only dependent products.

**Failure injection:** 429 with Retry-After, 5xx, interrupted transfer, corrupt JSON/PDF/archive, expired process lease, model OOM, malformed model output, timeouts, database lock, insufficient disk, native nonzero exits, stale output file, unexpected file writes, unavailable optional source, manual stop, and quota wait. Keep faults in task-owned fixtures and never disrupt other projects.

**Clean-room:** import/package check from a fresh environment and a working directory outside the repository, documented commands, UTF-8 Windows paths/console, local data/model cache availability, correct relative paths and environment requirements. Tests must not hard-code the benchmark machine's interpreter path.

## Scientific challenge families

Include wrong species/tissue/assay; proxy endpoint versus required endpoint; missing appropriate comparator; donors versus cells/runs; paired versus unpaired sampling; multi-dataset variable union without joint observation; false perturbational inference from a mention; background reference mistaken for the paper's result; quote present but meaning reversed; future work already answered later; explicit null versus unknown metadata; same paper/GEO/KG counted repeatedly; same subjects in multiple modalities; cross-species/units/estimand pooling failure; registry versus downloadable participant data; unknown access; publication versions/retractions; contradiction retained; no qualifying result; a useful partial result; hostile instructions in source text; mislabeled synthetic outputs; and manual model outputs masquerading as fresh runtime inference.

Use at least one held-out dataset with high semantic similarity but a genuine design mismatch. Evaluate question discovery on a source-selected sample, not only the generator's attractive outputs. Include questions/datasets lacking direct accession citation linkage, and report them separately from easy linked rediscovery cases.

## Evaluation design and labels

Before model/ranking tuning, freeze a scoped development protocol and an untouched study/publication-grouped holdout. Group repeated cohorts, duplicate articles, versions and paraphrases across split boundaries. Reserve a second topic or context for transfer. Temporal validation, if attempted, must use correctly dated snapshots; a modern KG cannot be leaked into a claimed historical novelty test.

Start with source-anchored challenge cases. Record each label's origin: `expert`, `source_determined`, `model_assisted`, or `unreviewed`, with annotator/method and source. Model-generated labels can be useful but are not expert gold. True expert calibration may remain pending; implement label import/export and versioned calibration without pretending it ran.

Measure extraction precision/recall where denominators are known; source-support and qualifier accuracy; entity-linking uncertainty; capability/group correctness; candidate recall; precision@k/nDCG or a justified ranking metric; invalid-top-match rate; no-fit/partial-fit behavior; lineage duplication; report traceability; latency/resource use. Full ranking evaluation needs judged candidates including negatives, not a list of known positives alone.

Choose numerical quality thresholds and material-improvement criteria before tuning from task risk, label provenance and baseline measurements. Record them and why. Do not tune on the final holdout. If thresholds are unrealistic or malformed, an independently justified protocol revision must be versioned before a new evaluation, with earlier results retained; silently lowering a failed gate is prohibited.

## Refinement and closeout

Execute at least two substantive passes per central worker and at least three integrated rounds. Fix failures, rerun affected cases and regression suites, compare to the incumbent, and reject regressions. Keep fresh holdout evidence separate from repeated development checks.

After all hardened gates pass and two consecutive development rounds show no material gain under the recorded criteria, perform the untouched holdout and independent closeout. If those reveal major defects, resume targeted repair and refresh compromised holdout cases. Stop when the remaining work is optional scope expansion or true diminishing returns, not merely when the first positive example appears.
