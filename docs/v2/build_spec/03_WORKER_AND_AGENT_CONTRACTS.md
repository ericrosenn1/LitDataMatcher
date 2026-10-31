# Workers, development agents, and parallel ownership

## Development roles

Use real Codex subagents when the installed environment provides them. Do not call OpenCode, Muse, or alternate coding-agent applications. These roles do not require ten concurrent agents. The lead may split or combine assignments while preserving independent review and clear ownership.

| Role | Primary assignment | Return artifact |
|---|---|---|
| Lead integrator | Outcome, scientific interfaces, dependency choices, integration, readiness | Compact task state, integrated commits, acceptance decision |
| Repository/reuse auditor | Current local work, migration map, donor licenses/APIs and small comparisons | Migration/reuse matrices with concrete evidence |
| Contracts/data-platform engineer | Typed records, migrations, local store, indexes, atomic work queue | Implemented contracts, round-trip and invalidation tests |
| Literature-acquisition engineer | PubMed/PMC/Europe PMC, local input parsing, section/version preservation | Live and replay parser/adapter checks |
| Semantic-evidence engineer | Model runtime, structured claims, source verification, normalized entities | Fresh-inference outputs, extraction challenge results |
| Dataset-catalog engineer | GEO and SRA/ENA study/sample/file interpretation, capabilities | Accession-level catalog, source-backed profiles and inspections |
| Question/matching engineer | Cross-document gaps, requirements, candidate retrieval, ranking | Executable matching plus positive/negative/unknown cases |
| Evidence-compilation engineer | External evidence import, lineage, compatibility, contradictions | Evidence bundles, integration demonstrations, dedup tests |
| Scientific evaluator | Evaluation protocol, independent challenges, label provenance, held-out testing | Error taxonomy, empirical scorecards, reproduced findings |
| Reliability/release evaluator | Process control, resource use, failure injection, clean delivery | Recovery/performance evidence and clean-install verdict |

The lead owns shared schemas, dependency lockfiles, integration branch, and publication decisions. Specialists may propose shared changes but should not collide on these files. Runtime workers are software components and do not have to resemble development-agent roles one-for-one.

## Minimum delegation contract

Each assignment has a task ID, parent milestone, verified base commit/source digest, specific input paths, allowed write paths, excluded resources, expected artifact/schema, acceptance tests, dependencies, stopping conditions, and concise return format. Send relevant scientific requirements as well as mechanical instructions. Do not ask a worker to infer the entire mission from a filename.

A returned patch/result states what changed; actual commands, exit codes and tests; unresolved failures; provenance or source versions; and integration instructions. Distinguish tests run from tests merely written. A worker cannot mark its own work as independently approved.

Assign independent read-heavy exploration concurrently. For write-heavy implementation create a verified Git worktree per active writer. Worktrees isolate checkouts, not OS access or model privileges; sandbox and directory permissions still matter. Do not supply secrets or unrelated project roots to workers. Do not edit `.git` pointer files to switch between Windows and WSL path conventions.

## Wave plan

**Bootstrap wave:** lead resolves and preserves workspace; auditor examines reuse; contracts engineer drafts implementable schema and state; evaluator freezes first source-based challenge cases. In parallel qualify one local runtime backend and start legal public acquisition.

**First working slice:** literature and dataset workers feed real artifacts into a minimal matcher/compiler. Data-platform and runtime work unblock both. Preserve a complete thin path early, then replace weak internals without losing it.

**Expansion wave:** improve sample profiling, semantic extraction, cross-document gaps, structured evidence, incremental updates and report usability. Tests and adversarial cases proceed alongside implementation, not only afterward.

**Hardening wave:** reviewers independently attack scientific assumptions, source lineage, data compatibility, recovery, offline mode and installed-package behavior. Writers repair bounded defects. Lead integrates and runs affected plus full milestone regressions.

**Closeout wave:** untouched holdout, clean-room install, final real run, readiness validator, pushed checkpoint, release folder and clear supervision stop state.

## Shared-state contract

There is one canonical job/lease store per build, outside separate worker checkouts. Worktree copies of `TASK_STATE.json` are not live synchronization primitives. Resolve the store from the verified build root. The lead writes compact versioned state snapshots; workers submit results through an atomic inbox/job table.

Use an OS-backed lock or correctly implemented transaction/lease with fencing for integration ownership. Record owner identity and process creation time as well as PID, branch/worktree and task. A stale timestamp alone cannot justify taking over an active writer. The scheduled supervisor uses this same lease and must never start a second lead on the same checkout.

Serialize necessary Git metadata/index changes and each database's writes. Readers may work concurrently from immutable snapshots. Use separate temporary output files per job and atomically promote validated output. Do not share mutable model sessions or overwrite common output names across workers.

## Independent scientific review

Give the critic the source material, generated claims, capability records and acceptance contract, but not only the author's persuasive explanation. Require concrete counterexamples with file/source locations and reproducible tests. Review at least interpretation/negation, group/control inference, dataset sufficiency, novelty updates, evidence lineage and ranking consequences.

When reviewers disagree, identify the precise factual or methodological disagreement, retrieve/check the source, and run the discriminating test. Do not hold indefinite consensus conversations or treat a vote among agents as scientific evidence. Document the resolution and the failing case used to settle it.

## Missing runtime features

Use only real available spawn/control tools and verified model settings. If subagents are unavailable, keep useful standalone deterministic work running, record the missing capability and provide the exact activation/continuation step. Sequential role-play may advance implementation but does not satisfy the independent multi-agent review gate. Do not claim a multi-agent build happened when only one agent wrote every assessment.
