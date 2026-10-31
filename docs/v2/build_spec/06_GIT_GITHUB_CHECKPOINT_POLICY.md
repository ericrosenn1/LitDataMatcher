# Git and GitHub: automatic safe checkpoints

## Authorized scope

Repository: `ericrosenn1/LitDataMatcher`.
Lead development branch: `codex/litdatamatcher-v2-build`, verified or created from the correct local starting state.

Routine local commits and pushes are explicitly authorized. They are checkpoints, not releases. Do not ask approval for every commit. No force pushes, rewriting already-published history, merges into `main`, repository visibility changes, public releases, or automatic uploading of large research data.

GitHub has previously been observed as public. Recheck the actual remote and visibility before the first push. A development branch in a public repository is public too. Authorization covers safe source code and compact project records, not secrets or unpublished unrelated research. Do not silently change privacy to solve a disclosure problem.

## Preserve the real starting state

Record the original HEAD, branch, staged and unstaged changes, and relevant untracked source. When creating a separate lead worktree, import those changes faithfully and verify content before committing. Preserve the original checkout/index where practical. One compact local recovery capture may be appropriate; do not make full-folder backup copies at each step.

Do not automatically replace a local branch with the remote default. Do not choose a checkout only because it has the newest modification timestamp. The June cleanup branch and historical paths are discovery clues. Untracked source can contain valuable work or credentials; inspect it before inclusion.

## Checkpoint behavior

At a coherent change or meaningful stage boundary, run relevant checks, inspect the actual diff and staged set, commit, and push. Before a likely context/session interruption, checkpoint meaningful work even if incomplete. WIP commits must accurately state failing/pending tests; tests need not all pass to preserve work. Never claim tests passed just because the commit succeeded.

If dirty work accumulates for hours, checkpoint at the next safe boundary instead of waiting for the entire system to finish. Do not create empty commits, commit every heartbeat, or commit a giant changing log every few minutes. The lead serializes integration. Worker branches may make isolated commits; the lead validates and integrates patches before updating the build branch. Preserve coherent history; cosmetic squashing is not needed during the run.

Use explicit project-scoped staging, not blind `git add -A` from an unknown root. Before push, check file type, size, source restrictions and credentials. A scan failure blocks that push, not unrelated local development. Remove secrets from a not-yet-pushed new change before publication; do not force rewrite an existing published branch automatically.

After push, compare remote branch SHA with the intended commit using a read-only remote check. A successful local commit is not a verified remote backup. Bounded transient retry is allowed. On authentication failure or rejected non-fast-forward update, preserve local work and record pending synchronization; inspect remote changes before integrating. Never force-push to suppress a conflict.

## Retention by artifact type

| Artifact | Git policy |
|---|---|
| Source, tests, schemas, configuration templates, dependency locks | Track |
| Final specification, compact state snapshots, decisions, worker scorecards, safe evaluation summaries | Track after content review |
| Run/source/model manifests containing hashes and permitted identifiers | Track compact sanitized versions |
| Raw publisher text/PDFs, human-level data, dataset files, large KG tables | Exclude unless an explicit small redistributable fixture has been reviewed |
| Model weights, embeddings, caches, databases, built indexes | Exclude; use manifests and lawful retrieval/rebuild commands |
| Raw tool transcripts, process telemetry, verbose stdout/stderr | Local rotated/compressed diagnostics, not routine Git content |
| Credentials, auth/config secrets, personal paths unrelated to the task, private account data | Never push |
| PID/lock/lease heartbeat files and mutable live state DB | Local only; publish compact semantic snapshots, not process churn |

Git replaces many versioned source-copy folders, not every scientific run manifest. It records committed source snapshots, not automatically which data/model/config generated a particular output. Keep that linkage explicit. A local repository still retains Git history; pushing is remote durability, not automatic disk cleanup.

## Avoid self-referential commits

Live local state may store the latest committed/pushed SHA. A tracked state snapshot should refer to the source commit it describes, or the previous checkpoint, without claiming to contain its own future hash. Do not create an infinite sequence of commits whose only change is 'latest commit'. Run IDs and artifact hashes provide stable joins.

## Cleanup

Remove only task-created completed worktrees after verifying results are integrated or explicitly rejected, there is no uncommitted/unpushed work to preserve, and no job still uses them. Use Git's supported worktree commands. Do not delete the original checkout, benchmark evidence, or other project worktrees. Keep a small number of live/recovery worktrees, not one forever for every minor task.
