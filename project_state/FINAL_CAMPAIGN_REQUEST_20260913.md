LITDATAMATCHER V2 FINAL AUTONOMOUS COMPLETION / CLOSEOUT CAMPAIGN

Execute this as one autonomous campaign from the current project state.

The objective is no longer to complete one bounded task. The objective is to take LitDataMatcher v2 from its present validated state to the maximum scientifically legitimate final state, including all remaining implementation, integration, validation, documentation, packaging, repository synchronization, and project closeout work.

Do not stop merely because the current NEXT_ACTION.md names one dependency. Do not ask for routine confirmation between stages. Do not return only a plan. Inspect the actual project, establish the remaining requirements, execute them, validate them, and continue until:

A. the complete intended project acceptance criteria are satisfied and the project can truthfully be marked COMPLETE, or

B. every machine-resolvable requirement is complete and the only remaining blocker is an irreducible external/human dependency that cannot legitimately be fabricated.

If B occurs, leave the project in a complete, resumable, review-ready state with exactly one explicit human/external gate and no avoidable engineering work left behind.

======================================================================
1. CURRENT VERIFIED STARTING POINT
======================================================================

Repository:
    ericrosenn1/LitDataMatcher

Canonical lead:
    C:\Codex\LitDataMatcher-v2\lead

Shared project data/state:
    C:\Codex\LitDataMatcher-v2\data

Development branch:
    codex/litdatamatcher-v2-build

Latest completed task:
    Exact DOI-absent PMID/PMCID cross-source literature reconciliation.

Functional commit:
    40921e58ca8a33e9de35847a116c324ec7d71500

Latest verified remote checkpoint:
    a4ee398cd8f30d275a8d151a46ffb13ef283392d

Latest bounded validation:
    targeted: 67 passed
    full suite: 346 passed
    final failures: 0
    final errors: 0
    final skips: 0
    protected alpha hash checks: 7 passed
    sealed holdout rerun: false
    live source retrieval during that task: false

Latest handoff:
    C:\Codex\LitDataMatcher-v2\lead\local\codex_review_handoffs\
    20260913T2026Z_pmid_reconciliation.zip

Latest reported next dependency:
    V2.4 real expert labels for existing review packets, followed by
    existing validation/agreement/adjudication machinery.

The previous task explicitly did not fabricate expert labels.

The hardened alpha is already complete and protected.
The final v4 holdout has already passed and is sealed.
Do not rerun or tune against the sealed holdout.

======================================================================
2. OPERATING MODE
======================================================================

Treat this as a final autonomous scientific-software completion campaign.

Continue validated work. Do not repeat bootstrap or rebuild completed stages merely because they exist.

Before substantial work, read the current versions of at least:

    AGENTS.md
    docs/v2/EXECUTION_OVERRIDE.md
    project_state/BUDGET_POLICY.md
    project_state/TASK_STATE.json
    project_state/NEXT_ACTION.md
    relevant docs/v2 build/specification files
    current acceptance/release documentation
    current review/calibration documentation
    most recent handoffs and receipts

Also inspect:

    git status
    git log
    branch/upstream state
    worktrees
    active processes
    validation artifacts
    release artifacts
    controller/state stores where relevant

Actual repository state and validated artifacts take precedence over stale prose fields.

Do not interpret an old RUNNING field as proof that a worker is currently active.

Preserve the user's original checkout and unrelated local modifications.

Use checkpoint/resume semantics throughout.

Do not restart an entire experiment because of a local failure.
Diagnose the failing stage, repair it, rerun only the affected stage when scientifically valid, and then run appropriate regression checks.

Preserve successful intermediate outputs.

======================================================================
3. EXECUTION POLICY AND RESOURCE POLICY
======================================================================

Read and obey the latest checked-in execution authorization and budget policy before starting reasoning workers.

The current repository policy and the user's latest persisted authorization supersede older instructions where they explicitly conflict.

Do not infer current authorization from an old handoff alone.

Verify the permitted model, effort, concurrency, and duty-cycle state from the current project policy/runtime before launching additional workers.

Never silently escalate to an unauthorized model, effort, paid API, credit pool, external billing route, or unsupported execution path.

Do not resume the corrective supervisor unless the current project policy explicitly permits it and its runtime model/effort is verifiable.

If reasoning concurrency is currently restricted to one worker, execute sequentially.

If multiple workers are currently authorized, use scoped Codex-native workers with:
    - isolated writer worktrees,
    - explicit ownership,
    - bounded tasks,
    - lead-controlled integration,
    - no duplicate writers on the same files.

The lead alone integrates and pushes.

During any mandatory reasoning idle interval, deterministic local processes that were already validly launched may continue, but do not violate the reasoning duty cycle.

If the interface cannot safely remain active through a required idle interval, create a durable exact checkpoint rather than violating policy.

Do not create a hidden or unsupported background automation merely to claim continuous execution.

======================================================================
4. FIRST TASK: RECONSTRUCT THE TRUE DEFINITION OF DONE
======================================================================

Do not simply execute NEXT_ACTION.md linearly.

Build a machine-readable final requirements-to-evidence matrix from:

    original finalized build/specification documents,
    later explicit execution overrides,
    current scientific contracts,
    acceptance criteria,
    current validated receipts,
    completed Phase 2 additions,
    release requirements,
    expert-review/calibration requirements,
    recovery/performance requirements,
    packaging requirements,
    repository state.

For each requirement classify it as:

    COMPLETE_VALIDATED
    COMPLETE_NEEDS_FINAL_REGRESSION
    MACHINE_COMPLETABLE_REMAINING
    HUMAN_EXTERNAL_REQUIRED
    OPTIONAL_FUTURE_VALIDATION
    SUPERSEDED
    NOT_APPLICABLE

For every classification record:
    - requirement source
    - exact evidence
    - validation receipt/path
    - relevant commit
    - remaining action if any
    - whether it blocks project completion

Do not weaken, reinterpret, or silently delete scientific acceptance criteria merely to obtain COMPLETE status.

At the same time, do not incorrectly treat an aspirational future validation item as a mandatory software-release gate if the governing specification does not make it one.

Resolve conflicting state documents by tracing the governing requirement and latest explicit policy.

Persist this matrix under an appropriate project_state or final audit location.

======================================================================
5. EXPERT-REVIEW / V2.4 DECISION
======================================================================

The current next dependency is reported as real expert labels.

Determine exactly what this means before declaring the whole project blocked.

Inspect:
    - V2.4 expert-review implementation
    - packet generation
    - scorecard/calibration-readiness implementation
    - agreement/adjudication machinery
    - original final acceptance specification
    - later documentation that introduced expert review

Answer internally and persist the result:

1. Are real expert labels a mandatory condition for the intended LitDataMatcher v2 final release/project COMPLETE status?

2. Or are they required only for an additional EXPERT_VALIDATED / CALIBRATED status beyond the machine-validated software release?

3. Can any current acceptance requirements legitimately use existing source-determined labels without claiming expert validation?

Do not invent, simulate, impersonate, synthesize, infer, or LLM-generate "real expert" labels.

Do not label records yourself and describe those labels as independent expert review.

Do not use multiple AI agents as substitutes for real expert reviewers.

If real expert labels are NOT mandatory for the intended final software acceptance:
    - finish the machine-validated project,
    - retain expert validation as PENDING_EXPERT_REVIEW,
    - accurately scope all claims,
    - do not let this optional external validation block software closeout.

If real expert labels ARE mandatory for project COMPLETE:
    - finish every other machine-resolvable item first,
    - validate the review machinery completely with synthetic/test fixtures only where appropriate,
    - generate a final human-review package,
    - make review instructions minimal and unambiguous,
    - include packet IDs, label definitions, adjudication instructions,
      expected input format, schema validation, blinded fields, and
      exact continuation command,
    - verify that supplied real labels can be ingested without code changes,
    - leave this as the only remaining blocker.

Do not stop at the expert-review gate while unrelated machine-completable work remains.

======================================================================
6. COMPLETE ALL MACHINE-RESOLVABLE REMAINING WORK
======================================================================

Using the requirements matrix, autonomously execute all remaining valid work.

This includes, where actually required by the governing specification:

    - incomplete Phase 2 source/integration hardening
    - identifier and provenance edge cases
    - lifecycle/retraction/version propagation
    - requirement compatibility handling
    - entity normalization
    - evidence dependence / deduplication
    - evidence compiler behavior
    - contradiction/incompatibility handling
    - review packet machinery
    - calibration-readiness logic
    - source-determined evaluation
    - cache refresh semantics
    - bounded pagination semantics
    - offline replay
    - recovery/resume behavior
    - deterministic invalidation
    - scale/performance instrumentation
    - dossier generation
    - CLI/API usability
    - packaging/installability
    - documentation
    - final regression and integration validation

Do not invent additional product scope just to keep working.

Only implement work that is:
    a) required by the governing specification,
    b) required to repair a demonstrated defect,
    c) required to make an existing promised feature valid,
    d) required for final reproducibility/installability,
    or
    e) required by final acceptance.

For every newly discovered defect:
    1. reproduce it with the smallest useful regression test,
    2. identify the affected contract,
    3. implement the narrowest coherent correction,
    4. run focused tests,
    5. run relevant integration/regression checks,
    6. update documentation/contracts where necessary,
    7. commit validated work,
    8. continue.

Do not silently change:
    scientific assumptions,
    endpoint definitions,
    compatibility semantics,
    evidence definitions,
    evaluation denominators,
    holdout membership,
    acceptance thresholds.

======================================================================
7. PROTECTED SCIENTIFIC ASSETS
======================================================================

The frozen hardened-alpha baseline and sealed final holdout are protected.

Never:
    - rerun the sealed holdout,
    - modify holdout records,
    - tune against holdout outcomes,
    - replace the holdout,
    - alter acceptance thresholds based on holdout results,
    - rebuild validated alpha artifacts without a demonstrated requirement,
    - redownload large validated datasets merely for convenience,
    - retrieve models again if validated local copies exist.

Use hashes and existing receipts to validate protected artifacts.

If a final acceptance check requires proving that protected outputs remain intact, use deterministic hash/status verification.

======================================================================
8. SCIENTIFIC INTEGRITY RULES
======================================================================

Preserve the project's fail-closed behavior.

Unknown remains unknown where the source does not support a stronger statement.

Do not infer:
    donor identity from metadata coincidence,
    biological sample count from technical run count,
    participant-data availability from registry presence,
    causal evidence from associative metadata,
    independence from records that may share underlying data,
    source completeness from bounded retrieval,
    expert validation from AI review,
    calibration from unvalidated labels,
    novelty from absence in a bounded candidate universe.

Corrected, retracted, versioned, duplicate, derivative, or otherwise dependent literature must retain the appropriate lifecycle/dependence semantics.

Preserve explicit provenance and source-scoped metadata through merging.

Do not use fuzzy title similarity as a hidden identity mechanism unless an existing explicit contract requires it.

======================================================================
9. VALIDATION STRATEGY
======================================================================

Do not rely on unit tests alone.

At appropriate checkpoints run:

A. Focused regression tests for changed behavior.

B. Relevant subsystem integration tests.

C. Full current test suite.

D. git diff --check.

E. Offline-cache replay checks.

F. Determinism checks where the project promises deterministic output.

G. Protected-alpha non-regression hash/status checks.

H. Existing adversarial cross-source regression suite.

I. Resume/recovery tests where required.

J. Performance-baseline comparison using the existing bounded fixture.

K. Packaging/build tests.

L. Clean-environment or clean-install smoke test from the final artifacts.

M. CLI end-to-end smoke test using permitted local/offline fixtures.

N. Final acceptance script/report.

Do not rerun sealed scientific evaluation simply to increase test coverage.

Capture machine-readable validation receipts with:
    timestamp,
    source commit,
    commands,
    environment,
    counts,
    hashes,
    status,
    scope,
    limitations.

A file's existence alone is not validation.
Inspect its schema/content and expected invariants.

Never print or record SUCCESS unless the final validation actually passes.

======================================================================
10. ADVERSARIAL FINAL REVIEW
======================================================================

Before declaring completion, perform an independent final review of the assembled project.

If current resource policy permits independent Codex review, use a non-writing reviewer or sequential independent review.

The reviewer should attempt to falsify the completion claim by checking:

    - requirement coverage
    - stale state documents
    - hidden test exclusions
    - false completeness claims
    - provenance loss
    - cross-source double counting
    - lifecycle bypasses
    - invalid identifier merges
    - unknown-to-negative coercion
    - calibration overclaim
    - expert-review overclaim
    - cache/offline behavior
    - package reproducibility
    - clean installation
    - recovery/resume behavior
    - Git cleanliness
    - release artifact integrity
    - protected holdout integrity

Any valid issue found must return to repair and revalidation.

Do not dismiss reviewer findings merely to finish.

======================================================================
11. FINAL RELEASE / PACKAGE CLOSEOUT
======================================================================

If the governing requirements permit final closeout without outstanding real expert labels, produce the final validated release artifacts required by the project.

At minimum, where consistent with the existing build specification:

    - final source state
    - wheel
    - source distribution
    - source/archive ZIP if specified
    - reproducible build metadata
    - dependency/environment documentation
    - machine-readable acceptance report
    - final requirements-to-evidence matrix
    - validation manifest with hashes
    - release notes
    - installation instructions
    - basic usage example
    - final project-state snapshot
    - independent final review receipt

Perform a clean-install validation from the built artifacts in a new temporary environment.

Validate that the package imports and the supported CLI entry points work.

Do not include:
    credentials,
    API keys,
    private user data,
    raw corpora,
    large downloaded datasets,
    model weights,
    unnecessary caches,
    verbose local logs,
    unrelated original-checkout modifications.

Use versioned output directories.
Do not overwrite prior validated releases.

======================================================================
12. GIT INTEGRATION POLICY
======================================================================

Use the existing development branch:
    codex/litdatamatcher-v2-build

Preserve unrelated existing user changes.

Do not:
    force-push,
    rewrite validated public history unnecessarily,
    merge to main without explicit existing project authorization,
    publish a public release unless the existing project specification
    explicitly requires and authorizes that action,
    expose local/private files.

WIP checkpoint commits are allowed when truthful and useful.

Before each push:
    - inspect diff,
    - inspect staged files,
    - ensure no secrets/large artifacts,
    - run required validation for the claimed state.

After each push:
    - verify the remote SHA,
    - verify the intended branch contains the commit.

The lead alone integrates and pushes.

======================================================================
13. PROJECT-STATE RECONCILIATION
======================================================================

As work progresses, keep project state truthful.

Do not preserve stale:
    RUNNING,
    pending task,
    worker-active,
    push-pending,
    next-action,
    test-count,
    model-policy,
    release-state

values simply because they are old.

Preserve historical evidence, but update current-state fields to reality.

At final closeout, reconcile at least:
    project_state/TASK_STATE.json
    project_state/NEXT_ACTION.md
    project_state/BUDGET_POLICY.md where applicable
    release/acceptance state
    relevant version/release docs

If project COMPLETE is scientifically justified:
    mark it COMPLETE using the project's established vocabulary.

If the only outstanding item is real expert review:
    do not mark expert validation complete.
    Record the software/machine-validation state separately from
    PENDING_EXPERT_REVIEW.

======================================================================
14. ONE-SHOT AUTONOMY RULE
======================================================================

Do not return to the user after each successful stage.

Continue autonomously through:
    audit
    implementation
    tests
    repair
    regression
    packaging
    clean-install validation
    final review
    Git checkpoint
    next remaining requirement

until reaching the actual stopping condition.

Routine uncertainty should be resolved by:
    project specifications,
    existing scientific contracts,
    tests,
    repository history,
    validated receipts,
    conservative fail-closed behavior.

Ask the user only when an unresolved decision would materially alter the scientific objective or when a genuine human/external input is indispensable.

Do not stop because a task took longer than expected if execution policy still permits continuation.

Do not manufacture work merely to remain active.

======================================================================
15. HUMAN-DEPENDENCY STOP CONDITION
======================================================================

If real expert labels are ultimately the sole mandatory blocker:

Finish EVERYTHING else first.

Then produce a directory such as:

    C:\Codex\LitDataMatcher-v2\data\phase2\v2_4_expert_review\final_human_gate\

containing, as appropriate:

    README_REVIEWERS.md
    REVIEW_PACKET.*
    LABEL_SCHEMA.json
    LABEL_DEFINITIONS.md
    reviewer-specific blinded packets if required
    example completed row clearly marked EXAMPLE ONLY
    validate_labels command/script
    ingest_labels command/script
    post_ingestion_validation command/script
    expected adjudication workflow
    HUMAN_GATE_STATUS.json
    remaining_requirements.json
    SHA256SUMS.txt

Validate the entire workflow using synthetic fixture labels that are explicitly marked synthetic and never used to make expert-validation claims.

The human should be able to provide real labels later without requiring another software-development round.

The project state should clearly distinguish:

    SOFTWARE_IMPLEMENTATION_COMPLETE
    MACHINE_VALIDATION_COMPLETE
    PENDING_REAL_EXPERT_LABELS

if that is the scientifically accurate final state.

Do not call this full expert validation.

======================================================================
16. FINAL STOP CONDITIONS
======================================================================

STOP CONDITION A: FULL PROJECT COMPLETION

Stop when:
    - all mandatory requirements are complete,
    - all required validation passes,
    - final artifacts are built and clean-install tested,
    - final adversarial review passes,
    - project state is reconciled,
    - validated commits are pushed,
    - remote branch SHA is verified,
    - no mandatory human/external dependency remains.

Then report COMPLETE.

STOP CONDITION B: IRREDUCIBLE HUMAN/EXTERNAL GATE

Stop only when:
    - every machine-resolvable mandatory task is complete,
    - all possible validation has passed,
    - release/package work not dependent on the human input is complete,
    - the human-review package is validated and ready,
    - there is exactly identified external input still required,
    - no coding task remains merely because it was deferred.

Then report MACHINE WORK COMPLETE / HUMAN GATE REMAINS.

STOP CONDITION C: EXECUTION/POLICY BLOCKER

If runtime authorization or resource policy prevents further valid execution:
    - checkpoint all work,
    - commit/push validated work where appropriate,
    - record exact unfinished stage,
    - preserve exact commands/inputs,
    - report the earliest safe continuation condition.

Do not falsely report completion.

======================================================================
17. FINAL OUTPUT TO TERMINAL / USER
======================================================================

At campaign end print a concise but complete final summary containing:

1. FINAL STATUS
   COMPLETE
   or
   MACHINE WORK COMPLETE / HUMAN GATE REMAINS
   or
   BLOCKED

2. Final functional source commit.

3. Final verified remote SHA and branch.

4. Final test and validation counts.

5. Final release artifact paths and SHA-256 hashes.

6. Clean-install validation result.

7. Acceptance report path.

8. Requirements-to-evidence matrix path.

9. Independent/adversarial final-review result.

10. Confirmation that protected alpha artifacts remained unchanged.

11. Confirmation that the sealed holdout was not rerun or tuned.

12. Expert-review/calibration status.

13. Any remaining external dependency, stated exactly.

14. Worker/supervisor status.

15. Whether the lead tree is clean and whether any unrelated local
    user changes remain intentionally preserved.

16. If COMPLETE, state explicitly that there is no additional
    machine-resolvable project work currently required by the governing
    specification.

Do not print SUCCESS or COMPLETE unless those claims are supported by the final receipts.