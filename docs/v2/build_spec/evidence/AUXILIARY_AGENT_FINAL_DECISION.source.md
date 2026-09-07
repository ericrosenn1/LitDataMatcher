# Auxiliary-agent final decision

## Decision

Do not add OpenCode to the multi-week LitDataMatcher v2 build. No production OpenCode model is selected.

The native Windows OpenCode CLI is now technically controllable: `C:\Users\eric\AppData\Roaming\npm\opencode.ps1`, version 1.18.29, supports the validated headless control path. This benchmark therefore tests task value rather than shortcut discovery. The existing Codex and Muse results were retained as frozen comparators and were not rerun. Muse remains excluded from the long-run build.

## Model-specific routing

| Model | Read-only analysis | Test generation | Small implementation | Production role |
| --- | --- | --- | --- | --- |
| `opencode/nemotron-3-ultra-free` | DO NOT USE | DO NOT USE | DO NOT USE | None; Task A was disqualifying, so B/C were not run |
| `opencode/big-pickle` | DO NOT USE | OPTIONAL | OPTIONAL | None; it did not beat the frozen Codex baseline on net constrained supervision cost |

Big Pickle is optional only for a detached, disposable task where a Codex lead has already budgeted independent test/contract review. It is not a general worker and is not authorized to edit shared or integration-bound worktrees.

## Frozen-task outcomes

### Task A — read-only repository analysis

Nemotron produced its report after 347.5 seconds but did not exit before the 420-second ceiling; its exact launched process tree was terminated at 430.2 seconds. It resolved only 19 of 21 cited source references and had at least seven material count/type errors. It was dropped before Tasks B and C.

Big Pickle produced a report in 103.8 seconds. All 115 cited source locations existed, but the report still mischaracterized `Evidence` as frozen and misstated multiple frozen hook-list counts. Its structure is useful, but lead correction of those factual assertions defeats the purpose of a low-supervision analysis worker. It is therefore not routed for analysis.

### Task B — test generation

Big Pickle generated 12 independently passing black-box tests in 159.1 seconds. They exercise real subprocess/file behavior and correctly contain the existing Windows console-encoding defect with `PYTHONIOENCODING=utf-8`. The suite is not directly reusable: it hard-codes the benchmark virtual-environment interpreter and does not cover the leading-whitespace comment case. The frozen Codex comparator was shorter, portable, fully passing, and needed no repair. Big Pickle is optional, not preferred.

### Task C — deterministic implementation

Big Pickle produced a 15-test implementation in 95.9 seconds. The suite passes, but independent strict probes show that non-string `source_id` values raise `AttributeError` rather than the required `ValueError`, and tuple `authorships` are accepted despite the list-only contract. The two missing tests allowed both defects through. Its private-helper decomposition also exceeds the frozen one-function scope. The frozen Codex comparator had 21 passing tests and exact contract compliance. Big Pickle is optional only for disposable drafts subject to full Codex repair/review.

## Net Codex effort

Provider-internal cost meters are not comparable to Codex effort, so cost was estimated from observable control work: task setup, launch/monitoring, independent validation, claim audit, repair required, and integration readiness.

Codex's frozen outputs required one bounded execution per task and no repair loop. OpenCode required native-launch plumbing, frequent monitoring, detailed correction of Task A factual claims, review of 12 Task B tests, and identification of two Task C defects missed by its own suite. Its apparent artifact speed did not reduce constrained Codex supervision enough to offset that review and repair tax.

## Substantive parallelism

The parallel control started a useful Codex CSV-CLI audit and a useful Big Pickle test-hermeticity audit in separate worktrees before either completed. Their reports were written 13.7 seconds apart; Big Pickle's report appeared 92.3 seconds after launch. Both used only their own worktree and neither modified production files.

This demonstrates functional additional parallel capacity, and the lead remained productive by validating candidate outputs while the jobs ran. It does not demonstrate useful *net* throughput: the Big Pickle audit needed correction for an unsupported test-execution claim and a non-deterministic `hash()`-based fake-embedding proposal. The parallel result is therefore PASS_WITH_LIMITATIONS, not a routing justification.

## Direct answers

1. **Which OpenCode model should be used?** None routinely. Big Pickle is optional only for isolated disposable drafts.
2. **Which exact categories should it receive?** At most detached test-draft generation and small deterministic utility drafts, with mandatory Codex review; not analysis.
3. **Which must remain Codex-only?** Source-grounded repository analysis, integration-ready test suites, strict-contract implementation, shared-worktree changes, and final validation.
4. **Does OpenCode reduce constrained Codex usage?** No, not in these tested conditions.
5. **Does it provide useful additional parallel throughput?** Functional overlap: yes. Net useful throughput after review: no.
6. **Does review burden negate the benefit?** Yes for production routing; only narrow disposable drafting remains optional.
7. **Should OpenCode join the multi-week v2 build?** No.

## Control and evidence limitations

Every run used native Windows `opencode.ps1`, explicit model, `--auto`, `--format json`, explicit isolated `--dir`, redirected stdout/stderr, and a 420-second ceiling. Raw JSONL shows normal terminal `reason: stop` events for Big Pickle; stderr was empty. The original controller did not persist the parent PowerShell numeric exit code before later polling, so semantic completion was established by terminal JSON event plus independent artifact validation rather than a stored process exit code. This controller-observability limitation is preserved rather than inferred away.

No candidate output was integrated. The source baseline, frozen task specifications, and prior Codex/Muse results remain unchanged.

**MUSE LONG-RUN BUILD: NO**
