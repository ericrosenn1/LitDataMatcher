# Master Codex task: build and harden LitDataMatcher v2

Specification 2.0.0 | 2026-09-07 | Repository identity: `ericrosenn1/LitDataMatcher`

## Assignment

Act as lead scientific-software engineer and integrator. Use actual Codex-native subagents to execute all ten workstreams, deliver a real functional local alpha early, and continue improving the workers and whole system until the hardened-alpha acceptance and closeout criteria pass. This is an implementation assignment. Architecture documents, installed dependencies, interfaces, mocks, and the legacy demonstration catalog are not completion.

LitDataMatcher must acquire scientific literature, interpret its meaning, represent evidence and unresolved questions, catalog current public datasets, encode their experimental capabilities, match questions to usable data, integrate relevant existing evidence across compatible and heterogeneous sources, and rank auditable research opportunities. Literature ingestion and dataset cataloging must also run independently and incrementally. The evidence compiler must feed back into question status and matching, not merely add citations to a finished report.

The goal is continuous **productive** work at the highest sustainable effort. There is no calendar deadline; days or weeks are acceptable. Do not deliberately slow execution, wait out a minimum duration, or consume inference watching a healthy script run. Use useful parallelism. Do not stop when the first pipeline works, and do not continue indefinitely after a hardened deliverable and diminishing-return criteria have been met.

## Read and establish authority

Verify the package manifest. Read `00_START_HERE.md` and companion files 02-16 plus `templates/BUILD_SPEC.json`. Use `15_SOURCE_NOTES_AND_DECISIONS.md` to distinguish verified evidence from historical clues and design choices. Preserve the originals as the frozen specification and copy the relevant package into the build branch's `docs/v2/build_spec/` after checking its contents for safe publication.

This specification replaces the older v1 task, including its no-commit/no-push rule and all earlier proposed OpenCode/Muse routing. Do not run an auxiliary-agent experiment again. The exclusion concerns development-agent applications, not ordinary open-source libraries or local inference models embedded in the delivered product.

Respect applicable system, sandbox, approval, and existing project instructions. Later explicit user changes can revise the task. Fix scientific implementation errors with a recorded decision and regression tests; do not silently lower acceptance gates or change a held-out benchmark to fit a failing result.

## Authority to act

The user authorizes project-scoped source edits, substantial refactoring, isolated environments, permitted public-data/model retrieval, tests and benchmarks, bounded repairs, local Git branches/worktrees, checkpoint commits, and pushes to a dedicated development branch on the verified existing remote. Do not ask for routine approval between stages.

Do not force-push, rewrite existing published history, merge into `main`, create a remote release, change repository visibility, delete pre-existing work, purchase services, change billing, expose credentials, or upload raw papers/data/model weights. Do not take over resources or files belonging to other research projects. Existing credentials do not establish a new API spending budget. Use the supported Codex account normally for development; do not extract subscription tokens to power the application.

## First actions

1. Resolve the real local project. Inspect the active workspace, Git root, branch/remotes, staged/unstaged/untracked content, current instructions, ongoing jobs, and newer local changes. Verify historical candidates in file 12 only if needed. Never build from a disposable benchmark source snapshot or overwrite local development with GitHub main.
2. Record the initial state once. Preserve staged and unstaged patches and relevant untracked source, excluding secrets and bulk data. Prefer a separate lead worktree based on the correct local commit plus safely imported uncommitted work; leave the user's original checkout/index intact. Make one compact safety capture where necessary, not repetitive directory backups.
3. Establish the lead branch `codex/litdatamatcher-v2-build`, or safely resume that branch after verifying it belongs to this build. Inspect the remote before pushing. Create a source-preservation checkpoint and verify the pushed ref. A push failure must not erase local work or cause a force push.
4. Discover actual OS, Python, Codex client/subagent controls, supported reasoning settings, current allowance visibility, RAM, CPU, GPU/VRAM, free disk, local model backends, and shell permissions. Keep native Windows control unless a measured dependency needs WSL. Do not rewrite `.git` files to translate paths.
5. Run the existing bounded regression baseline and example once where prerequisites permit. Record known defects rather than masking them. Create the compact work queue, shared integration lease, and initial task state. Verify a legitimate runtime-model path early so missing inference does not appear only at delivery.
6. Delegate independent starting-state/reuse, evidence/data-contract, and evaluation work. Implement a thin real end-to-end slice while other workers build compatible modules. Do not spend the whole run perfecting architecture before trying integration.

## Ten workstreams to complete

| ID | Workstream | Operational result |
|---|---|---|
| W01 | Architecture and scientific contracts | Versioned typed records, module boundaries, field-level provenance, migrations, and interfaces exercised in code |
| W02 | Repository preservation and migration | Useful local work retained; one authoritative package path; legacy/demo data separated; regression behavior accounted for |
| W03 | External-component qualification and reuse | Actual donor/source/license/dependency checks, small executed comparisons, selected integrations and documented alternatives |
| W04 | Local data plane | Cached snapshots, normalized records, local search/indexes, durable job state, incremental updates, atomic writes and offline operation |
| W05 | Literature acquisition and parsing | Real PubMed/PMC or Europe PMC access, section-preserving structured/PDF parsing, DOI/PMID/version reconciliation |
| W06 | Semantic evidence extraction | Fresh application-run model inference, entity resolution, structured claims, source-span and meaning checks |
| W07 | Question and requirement discovery | Automatic gaps plus explicit-question mode, dated unresolvedness, formal measurable/design requirements |
| W08 | Dataset acquisition and capability profiling | Real accession-level GEO and sequencing-repository metadata, sample/group profiles, file inspection and explicit missingness |
| W09 | Evidence compilation | Literature plus external structured evidence, dependence-aware bundles, contradictions, justified mappings, numerical integration only where valid |
| W10 | Matching, evaluation, ranking and delivery | Indexed candidate retrieval, experimental-fit checks, interpretable ranking, benchmark/tuning machinery, review report and tested user commands |

All ten must have implemented paths and executed evidence. Subcapabilities that require absent expert labels must be reported accurately; implement their machinery rather than fabricating labels or claiming calibration happened.

## Agent organization and execution

Use one lead and the specialist/reviewer roles in file 03. Roles are responsibilities, not a demand for ten active sessions. Spawn genuine subagents, record their actual identifiers when exposed, give writers isolated worktrees and scoped ownership, and have the lead serialize integration and shared-schema changes. Reviewers evaluate other agents' work and reproduce important findings. Do not manufacture independent reviews.

Use Astra High as the initial substantive-work preference, with higher effort for consequential decisions when supported and sustainable. Do not invent configuration names or claim a setting changed without verifying it. Prefer Standard speed. Ultra is excluded. Adjust concurrency from observed productive throughput, workload dependencies, local pressure, and available allowance. Do not silently route difficult scientific work to weaker models to stay busy.

Run deterministic downloads, validation, parsing, indexing, inference batches, and tests as properly supervised jobs. Share immutable caches; serialize necessary database/Git writes. Protect other workloads and preserve RAM/VRAM headroom. Apply idle-aware scheduling with hysteresis if supported; do not stall the build while engineering a sophisticated resource governor.

## Scientific requirements that cannot be substituted away

- Evidence records must preserve source identity, raw locator, interpretation, biological conditions, measurement type, and lineage. Embeddings are an index, not the authoritative evidence object.
- Semantic similarity cannot rescue a failed indispensable experimental requirement. Unknown metadata is neither verified compatibility nor verified absence.
- Dataset availability, design compatibility, and actual statistical answerability are separate assessments. Study-level metadata must not invent usable controls or donor counts.
- The compiler must consider indirect, contradictory, and cross-modality evidence under explicit roles and mappings. Do not simply concatenate matrices or count every source record as an independent vote.
- A paper, its GEO submission, and a graph entry copied from it share lineage. Different assay types are not automatically independent if they reuse the same subjects or experiment.
- Existing related biology may resolve, narrow, contradict, or leave a question open. Record search coverage and time; do not infer global novelty from one future-work sentence or missing search hits.
- Distinguish downloaded metadata from analysis actually run on data. A ranked proposal is not a completed downstream experiment.
- The delivered application must execute fresh semantic extraction and matching on a new input without Codex authoring the output by hand. It must support a qualified local runtime model and offline reruns after resources are cached. Do not use OpenCode/Muse or an unofficial subscription bridge as its backend.
- Evaluate held-out sources, hard negatives, and the final question-dataset-evidence object. Separate source-determined, expert, model-assisted, and unreviewed labels. Never fabricate independent human validation.

## Iterative development and stopping

Build and validate the first functional slice, publish a compact interim local report, and continue. Every central worker receives at least two substantive evaluation/refinement passes; the integrated system receives at least three distinct rounds, including independent scientific challenge and clean-install/offline/recovery testing. A pass may reject an unnecessary code change if executed evidence supports the current implementation. Repeatedly rerunning identical tests does not count as improvement.

Drive fixes by error severity and scientific consequence. Add regression cases, compare against the current incumbent, preserve successful intermediate work, and rerun only affected stages. Keep a held-out set separate from tuning; do not report repeatedly exposed cases as fresh validation. Model training is selective and justified by observed errors, not mandatory for its own sake.

File 08 defines mandatory gates. File 09 separates `FUNCTIONAL_ALPHA_AVAILABLE` from `HARDENED_ALPHA_READY`. Expanded real-data and transfer checks, real fresh-input operation, and the agreed refinement must be complete before closeout. No unresolved critical/high issue affecting declared functionality may be papered over. When gates pass and two consecutive development evaluation rounds yield no material improvement under predeclared criteria, complete the final untouched-holdout/independent checks and deliver. Do not invent new scope to avoid finishing.

## Git, state, supervision and interruption

Commit and push coherent safe checkpoints, including valuable WIP with truthful test status. Keep `main` unchanged. Source code, tests, compact decisions and manifests belong in Git; bulk corpora, credentials, models, indexes, transient PID/lease state and verbose logs do not. Track the latest pushed checkpoint in local state rather than recursively committing a file just to record its own commit hash.

Maintain a compact task state, exact next action, append-only meaningful decisions, and per-run manifest. Use one shared live job/lease store across worktrees; versioned state exports alone do not coordinate writers. No per-minute handoff proliferation.

Prepare the corrective supervisor in file 10 early and test it against isolated recoverable faults. It must leave healthy work alone, respect manual pauses and quota waits, resume only via a demonstrated authorized control path, and use the same integration lease. Enable a local schedule only when access and actual repair/resume are verified. If the installed client cannot create or execute it, provide the precise remaining local UI step and continue the main build. An hourly report-only task is not the requested supervisor.

At a rate limit or session interruption, checkpoint, record a verified reset time when available, preserve safe deterministic work, and provide an executable continuation point. Do not bypass limits or imply a stopped session is still working. A scheduled prompt does not guarantee continued execution during app shutdown or provider exhaustion.

## Required delivery

Deliver the tested local package/source, locked environment, optional local model setup, configuration, real-data HTML review report, source/data/model manifests, worker scorecards, comparative benchmark results, independent reviews, machine-generated acceptance report, and exact native PowerShell launch/resume/test commands. Open the local output/report when useful.

Report three separate outcomes: product readiness, development supervision/recovery readiness, and scientific calibration status. Expert ranking calibration may remain pending with transparent heuristic scores; a missing real inference path or failed required product gate may not.

Final message: product version and source commit; what was retained/replaced/reused; which workstreams ran; real coverage and test counts; remaining limitations; last pushed branch/SHA; active/stopped supervisor status; local report/delivery paths; tested run commands; and why refinement stopped.

**Begin now. Do not end after the audit, ask permission to start each next phase, or substitute a scaffold for the requested functioning and hardened tool.**
