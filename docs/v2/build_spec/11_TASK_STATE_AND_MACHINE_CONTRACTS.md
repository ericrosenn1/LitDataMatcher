# Machine contracts and durable state

`templates/BUILD_SPEC.json` contains fixed project choices, workstream IDs, milestone coverage floors and review minima. It is the numeric counterpart of files 08 and 09. The JSON files are templates/schemas, not a claim that a build or schedule has started.

## State model

Use one canonical local live state/job store. Choose a compact SQLite table set or an equally safe durable mechanism; the lead owns promotion/integration. Export `TASK_STATE.json` and `NEXT_ACTION.md` only on meaningful changes, interruption and checkpoint, not every process poll. Worker checkouts may read those exports but do not each own a competing live controller.

The state schema captures build/spec identity, product/runtime/automation/calibration status, local source authority, current phase, progress, unresolved blockers, artifact references, source commit, and job pointers. Detailed PIDs/leases/counters may remain in the local store. Credentials, long transcripts and biological datasets do not belong in state exports.

Operational state values include BOOTSTRAP, RUNNING, WAITING_FOR_CAPACITY, WAITING_FOR_ACCESS, PAUSED_BY_USER, BLOCKED, COMPLETE and TEMPLATE_NOT_RUNNING. A manual pause cannot be automatically converted into RUNNING by a heartbeat. A new context must verify real processes and input/artifact versions before trusting a stale RUNNING value.

## Templates

- `templates/TASK_STATE.template.json` and `.schema.json`: initialize only a new build, with actual resolved paths and version/commit. Never overwrite existing validated state on resume.
- `templates/RUN_MANIFEST.template.json` and `.schema.json`: source/model/config/code and artifact lineage for one actual run. `execution_status=NOT_RUN` is the template default.
- `templates/ACCEPTANCE_REPORT.template.json` and `.schema.json`: one record per required product gate, with executed evidence. Template gates are NOT_RUN and product is NOT_READY.
- `templates/BUILD_CONFIG.example.json`: local implementation settings and auto-discovery placeholders. These are LitDataMatcher design defaults, not configuration keys to paste into Codex's own config file.

Validate schemas and business rules. A structurally valid manifest does not verify file hashes, actual source coverage, genuine test execution or scientific judgment. The implemented release validator must check those separately and reject stale evidence for a changed code/model/config version.

## One source of operational truth

`NEXT_ACTION.md` should contain the exact next safe command/assignment, its cwd, prerequisites and expected validation. A command is only 'tested' after actual execution; use 'proposed' beforehand.

Record the actual common state root and Git root so worktrees and supervisor share coordination. Make state writes atomic and recovery from a truncated write explicit. Use an integration lease and per-job ownership to prevent duplicate task dispatch or two agents taking control simultaneously.

Do not recursively commit 'latest SHA' updates or auto-increment package versions on every poll. Commit IDs and input hashes identify meaningful stages. One short decision ledger records substantive changes, not every shell command.
