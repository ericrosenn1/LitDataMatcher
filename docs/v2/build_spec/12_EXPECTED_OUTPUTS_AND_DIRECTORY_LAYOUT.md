# Workspace discovery and delivery layout

## Repository discovery

Start with the local folder the user opened. Confirm it is the LitDataMatcher source, not a review snapshot. These historical locations are useful search candidates, not authoritative current paths:

```text
C:\Users\eric\Documents\Codex\2026-05-31\github-plugin-github-openai-curated-can\work\LitDataMatcher
C:\Users\eric\OneDrive\AI project\LitDataMatcher-main\LitDataMatcher-main
```

An older local cleanup branch was reported as `cleanup/repo-layout-20260615`, with layout/provenance work beyond remote main. Verify actual branch/content and preserve uncommitted changes. Remote main inspected during package preparation was `48ef6580efccf55578dd865cb7154cfa34c5a872`; it is not a directive to reset to that commit [S06].

Search those candidates and sensible bounded locations under Documents/Codex, OneDrive/AI project and Downloads only when the active project is missing. Compare remotes, source content, commits, state files and local changes. Do not select between diverged substantive checkouts solely by file timestamps. Ask once only if authority cannot be resolved safely.

Explicitly exclude disposable trees named like `LitDataMatcher_muse_delegation_benchmark_20260907`, auxiliary benchmark source snapshots, and the smoke-test folder as production roots. They contain useful evidence, not the current app.

## Storage policy

Keep one source/lead workspace plus the necessary active writer worktrees. Select a data root based on actual free local disk and existing project organization; do not assume a D: drive exists or is unused. Prefer bulk caches outside OneDrive/Git synchronization. Do not move existing personal files or other projects to create space.

A possible layout, adaptable to the real project:

```text
<verified Git root>/
  litdatamatcher/                authoritative application package
  tests/                        unit, contract, integration and regression tests
  docs/v2/
    build_spec/                 this frozen specification, copied once
    STARTING_STATE.md
    ARCHITECTURE_V2.md
    CURRENT_REPO_MIGRATION_MATRIX.tsv
    EXTERNAL_COMPONENT_REUSE_MATRIX.tsv
    EVALUATION_PROTOCOL.md
    WORKER_SCORECARD.tsv
    DECISIONS.md
    KNOWN_LIMITATIONS.md
  project_state/
    TASK_STATE.json              compact versioned milestone export
    NEXT_ACTION.md
  benchmarks/                   permitted fixtures, labels, compact summaries
  local/                        ignored live state, logs, leases and diagnostics
  <existing legacy locations>   retained/migrated after inspection

<shared external data root>/
  sources/                      immutable raw snapshots, text, metadata
  models/                       versioned qualified local weights/tokenizers
  normalized/                   versioned scientific records
  indexes/                      rebuildable local retrieval indexes
  runs/<run_id>/                real run results and manifests
  controller/                   one shared live job/lease store if outside repo
  worktrees/                    optional location for active task worktrees
  releases/<alpha_version>/     validated user-facing local deliverable
```

This is an example, not a command to reshuffle a working repository to match labels. Persist actual roots in one configuration and state pointer. Use safe relative paths in portable manifests and retain resolved local paths only where needed.

## Required run outputs

For an actual integrated run, emit a run manifest, document/source catalog, structured claims, normalized dataset/capability records, questions and requirements, evidence bundles, matches, review labels/export, and HTML report. JSONL or equivalent rich validated tables retain nested provenance; CSV is a convenience view, not the sole authority.

Key filenames or documented equivalents:

```text
RUN_MANIFEST.json
claims.jsonl
questions.jsonl
datasets.jsonl
evidence_bundles.jsonl
matches.jsonl
review_sheet.csv
report.html
WORKER_SCORECARD.tsv
INTEGRATED_BENCHMARK.json
FINAL_SCIENTIFIC_REVIEW.md
FINAL_ENGINEERING_REVIEW.md
ACCEPTANCE_REPORT.json
RELEASE_READINESS.md
```

Do not create one new folder of handoffs for each subagent action. Store concise per-attempt evidence under the relevant run; rotate verbose logs, keep decisive failure cases and final benchmark summaries, and remove redundant transient copies only after validation.

## User delivery

Ship a versioned installable package/source bundle, no secrets or bulk datasets, with `START_HERE.md`, tested Windows PowerShell setup/run/stop/resume commands, a configuration template, environment lock, notices and retrieval manifests, a real-data report and acceptance results. Version the application from its actual local state; specification version 2.0.0 does not force application semver 2.0.0.

Test the package outside the source checkout. Do not confuse editable-source success with installed-distribution success. Confirm required data/model files are available or retrieval instructions work. Automatically open the report/folder where supported and useful. The final response must include precise local paths and remote build-branch SHA, not merely 'done'.

Any nontrivial PowerShell delivered later must use strict mode, terminating errors, checked native exit codes, guarded process cleanup, safe argument handling, validated outputs, a concise terminal result and no false SUCCESS. Account for the actual PowerShell version. Wrap pasteable multi-statement try/catch/finally code in one script block so interactive pasting cannot detach `finally`.
