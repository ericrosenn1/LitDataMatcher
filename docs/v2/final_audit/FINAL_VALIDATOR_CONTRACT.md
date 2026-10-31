# Final campaign machine acceptance contract

The installed entry point is `python -m litdatamatcher.final_campaign_acceptance`.
The checkout wrapper is `python scripts/v2/validate_final_campaign.py`.
Both read retained evidence without acquisition, model inference, test execution,
remote Git queries, or sealed-holdout execution. Local Git metadata and the seven
protected files are read only. JSON reports retain one PASS/FAIL result per gate,
errors, and actual JUnit testcase counts.

```powershell
C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe -B -m litdatamatcher.final_campaign_acceptance --example-ledger --out LEDGER_EXAMPLE.json
C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe -B -m litdatamatcher.final_campaign_acceptance --snapshot-source C:\Codex\LitDataMatcher-v2\lead --out SOURCE_FINGERPRINT.json
C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe -B -m litdatamatcher.final_campaign_acceptance --ledger FINAL_LEDGER.json --mode technical --out TECHNICAL_ACCEPTANCE.json
C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe -B -m litdatamatcher.final_campaign_acceptance --ledger FINAL_LEDGER.json --mode closeout --out FINAL_ACCEPTANCE.json
```

Exit 0 means the requested mode passed. Exit 1 means at least one required gate
failed. Malformed, missing, stale, or contradictory evidence fails closed.
`technical_readiness` is independent of `operations_closure`; a technically ready
assembly may still await final state, handoff, or Git closure. This validator does
not require a campaign controller to say COMPLETE before checking the software.
Expert validation remains `PENDING_EXPERT_REVIEW` and heuristic ranking remains
`UNCALIBRATED_HEURISTIC`; these are independent of software readiness.

## Artifact references and source binding

Every ledger artifact reference is an object containing `path`, lowercase raw
byte `sha256`, and integer `size_bytes`. Paths may be absolute or relative to the
ledger directory. These semantics also apply to nested command, case, and review
references. Matrix source/evidence paths may instead be relative to the source
checkout and may omit size when using the original audit schema. Native run
artifacts resolve relative to their native manifest; benchmark artifacts resolve
relative to the benchmark receipt. Paths and hashes are checked, not just listed.

`source` contains `root`, full `functional_commit`, and `fingerprint` (an artifact
reference to a `final_functional_source_v1` snapshot). `source_snapshot(root)`
returns `source_root`, `source_commit`, `files`, `digest`, and
`dirty_functional_paths`. `files` maps relative functional paths to SHA256 after
CRLF-to-LF conversion. `digest` uses `data_plane.digest(files)` (sorted, compact,
UTF-8 JSON). This is deliberately separate from raw artifact hashes.

Functional scope is tracked and untracked Python/JSON/JSONL/lock files under
`litdatamatcher/`, `tests/`, and `scripts/v2/`, plus `pyproject.toml`. Cache
directories are excluded. New untracked functional files count. Current files
must exactly match the snapshot and the declared committed tree, be committed,
and have that functional commit in current HEAD ancestry. Documentation-only
commits do not invalidate successful execution on identical functional bytes.

## Ledger schema `final_campaign_ledger_v1`

`--example-ledger` emits a complete top-level template with deliberate invalid
placeholders. Required fields are:

| Field | Required value |
| --- | --- |
| `source` | Source root, functional commit, fingerprint reference |
| `matrix` | Final 74-row requirement matrix reference |
| `junit` | List of `scope`, `xml` reference, `command` reference |
| `distribution` | Version, wheel, sdist, source-archive references |
| `clean_install` | Nonempty list of executed command references |
| `offline` | Nonempty list of executed offline command references |
| `cases.manifest` | Final source-assisted cases manifest reference |
| `acquisition` | Receipt, plan, offline replay, membership, partition-receipt references |
| `omics` | Qualification receipt and declared protocol references |
| `scale.receipt` | Native real-metadata benchmark receipt reference |
| `independent_review` | Current-source-bound independent review reference |
| `protected_alpha` | Exactly seven named references, with pinned authority hashes |
| `git` | Branch, remote, remote branch, checkpoint, remote-observation command reference |

The optional top-level `expert_validation` can only be
`PENDING_EXPERT_REVIEW` for this campaign.

## Executed command schema `final_command_receipt_v1`

Required fields are `argv` (nonempty string list), absolute `cwd`, timezone-bearing
`started_at` and `finished_at`, integer `exit_code`, full `source_commit`,
`source_fingerprint` (the functional snapshot digest), explicit
`dirty_functional_paths` list, `stdout` reference, and output/input references
where applicable. Commands must actually have been executed by the producer.
The validator verifies exit results and retained output bytes; it does not turn
handwritten command descriptions into execution evidence.

Successful commands require exit 0. Test commands retain their actual exit and
JUnit failures. A command recorded while files were dirty may be accepted when
its recorded functional digest exactly equals the final committed bytes. The
dirty paths remain visible. A receipt claiming a clean source tree must also
match the bytes at its recorded source commit.

JUnit scopes are `full`, `targeted`, and `integration`. Full plus focused or
integration execution is required. XML references must also occur in command
outputs. Every actual testcase is counted, duplicate testcase identities are
rejected, and XML-declared totals must equal testcase-derived totals. Failures,
errors, skipped tests, a failed command, and empty collection fail. Selective
pytest flags and file/node selection cannot be relabeled as the full suite.

## Packages and installed execution

Wheel, sdist, and source archive are reopened without extraction. ZIP CRCs, path
safety, duplicates, link/special entries, uncompressed sizes, wheel RECORD hashes
and sizes, package name, and wheel/sdist/ledger version agreement are checked.
Runtime caches, model weights, and selected private/runtime artifact classes are
rejected. Package Python bytes must match functional source; the source archive
must contain the full functional snapshot. This is an integrity gate, not a
general-purpose malware scanner.

Clean-install command `inputs` must bind the delivered wheel. Its stdout is JSON:

```json
{
  "schema_version": "final_install_observation_v1",
  "package_file": "ABSOLUTE_ENV/site-packages/litdatamatcher/__init__.py",
  "package_version": "0.3.0",
  "sys_prefix": "ABSOLUTE_ENV",
  "executable": "ABSOLUTE_ENV/Scripts/python.exe",
  "cwd": "ABSOLUTE_DIRECTORY_OUTSIDE_SOURCE",
  "cli_results": [
    {"name": "litdatamatcher", "argv": ["ACTUAL_EXECUTABLE", "--help"], "exit_code": 0, "stdout": {"path": "help.log", "sha256": "HASH", "size_bytes": 1}},
    {"name": "litdatamatcher-v2", "argv": ["ACTUAL_EXECUTABLE", "--help"], "exit_code": 0, "stdout": {"path": "v2_help.log", "sha256": "HASH", "size_bytes": 1}}
  ]
}
```

The producer must execute the installed interpreter and both CLIs. The validator
checks reported exit codes, executable/import locations, retained stdout, installed
version, and every package file against delivered wheel bytes. An import from the
source checkout fails even if version text agrees.

## Offline and cases

An offline command's stdout has schema `final_offline_observation_v1`,
`network_control={blocked_probe_count:>=1, unexpected_requests:0}`, and `fresh`
and `replay` objects. Each has `origin`, `manifest` reference, and comparable
scientific `payload` reference. Origins are respectively `fresh_local_inference`
and `cache_replay`. Native `RUN_MANIFEST.json` files retain every artifact hash and
size, successful execution, model ID/revision, and exact inference-origin counts.
Retained `inferences.jsonl` rows must show local-only execution. Fresh/replay
input, model, implementation, and configuration fingerprints must agree, as must
the explicitly selected comparable scientific payloads. The outer executed
command retains the active network guard observation. Original fresh inference
provenance must not be rewritten as if it occurred on a later source revision.

`final_cases_manifest_v1` contains source commit/fingerprint, an executed `command`
reference, `data_origin:"real"`, a `dataset_catalog` JSONL reference, at least six
`cases` across at least two domains, and a `source_locators` list. Each case has
`case_id`, `domain`, one JSON dossier reference, and `run_manifest` reference.
Every dossier must be an output bound to the executed case command and occur in
its native run's scientific_dossiers.jsonl. Each source-locator entry has
`locator` and `artifact` reference.

Dossiers must satisfy the product dossier schema, use distinct question/candidate
pairs, retain source evidence IDs, source locators, scoped novelty, and pending
expert review. Candidate IDs/fields must agree with the retained dataset catalog,
whose IDs must belong to the verified acquisition/omics inputs.
Every evidence locator resolves to a retained hashed artifact. A supplied source
span must occur in that source; direct experimental-evidence roles require a
span. JSON/JSONL string fields are decoded for this comparison; provided start/end
offsets must reproduce the span in a retained source text field. Deterministic
source passages must remain background, NOT_ASSERTED, inconclusive and explicitly
not an answer to the question. They require no invented model claim object.
Synthetic review and calibrated probability claims fail.

Native FAIL status always fails. PARTIAL is accepted only for disclosed
source_guard rejections, while every retained evidence span is independently
checked. Accepted source-validated claims may coexist with unasserted background
context and rejections; no all-context restriction is imposed. Inference failures
cannot be converted into successful case coverage. The acceptance report retains
native failures, coverage, inference counts, and accepted-claim/dossier counts.
The six-case, two-domain floor is the selected final campaign execution plan; it
does not redefine frozen alpha G12 (failure recovery) or equate context dossiers
with structured claim extraction accuracy.

## Native acquisition, omics, scale and recovery

Expanded acquisition uses its actual `ACQUISITION_RECEIPT.json`,
`PREDECLARED_PLAN.json`, `OFFLINE_REPLAY.json`, `corpus/membership.jsonl`, and the
13 explicit partition receipt references. Each partition receipt binds sibling
`records.jsonl` by `records_sha256`. Original request snapshots retain URL, object
path, SHA256, byte size and HTTP status; abbreviated offline snapshot events must
resolve to their original request metadata. Counts, unique IDs, repositories,
domains, request/record bounds, source snapshots and membership are recomputed.
The frozen campaign targets are at least 1000 literature records, 300 dataset
study IDs, two repositories and three domains, or higher declared plan values.
Offline comparable hashes are recomputed using the native sorted JSON encoding,
removing only recursive `cache_status`. This verifies retained replay evidence;
it does not reacquire or rerun the original adapter.

`omics_qualification_v1` binds the declared protocol, actual artifact, PRIDE and
Metabolomics Workbench study IDs, raw snapshots, offline receipt, and metadata-only
scientific status. Study identity counts are not treated as donor/independent-unit
counts. Normalized partition digests are recomputed in the retained raw source's
record order, with matching raw study IDs and partition counts.

`phase2_real_metadata_benchmark_v1` is consumed directly. All native plan, input,
source-file and artifact hashes are checked. Three actual scale points must include
the full final input. Catalog SQLite files are opened with `mode=ro&immutable=1`;
current IDs, version counts and payload digests are compared with actual input.
Cache hit denominators and declared latency/memory bounds are checked. Source
metadata/reference mappings independently reconstruct each pair's field judgment
and label. Every method's complete candidate universe, precision/recall/nDCG/RR
denominators and values are recomputed; unknowns remain unjudged. Real lexical,
pretrained semantic, compatibility and ablation baselines are required. Observed
fits cannot be excluded, and mismatches/unknowns cannot be promoted.

Recovery must retain separate child processes with observed exit codes 71 then 0,
partial committed IDs, exact resumed-only-missing IDs, unchanged input hash, and
final catalog/version/payload equality. Summary booleans cannot replace these
observations. Compiler context throughput and source-determined compatibility
metrics do not establish scientific-claim extraction accuracy or calibration.

## Matrix, review, preservation and operations

The exact 74 IDs from `REQUIREMENT_SCOPE_REVIEW.json` are pinned in code. Every row
has a governing criterion/source and nonempty hashed evidence. ALPHA, PHASE2 and
FINAL implementation rows remain mandatory and require `COMPLETE_VALIDATED`,
nonblocking dispositions. Optional/component/history dispositions use the
original vocabulary. EXPERT rows are `OPTIONAL_FUTURE_VALIDATION`, nonmandatory,
nonblocking and pending.

`FINAL.MATRIX`, `FINAL.ACCEPTANCE`, and `FINAL.RECEIPTS` may retain
`COMPLETE_NEEDS_FINAL_REGRESSION` with `validation_contract:"THIS_VALIDATOR"` and
nonblocking status to avoid requiring their own output beforehand. Final STATE,
PUSH, CLOSEOUT, FINAL_REPORT and SUPERVISOR rows can remain unfinished without
changing technical readiness, but block operations closure.

Independent review schema `final_independent_review_v1` contains `status:"PASS"`,
source commit/fingerprint, different nonempty `reviewer_id` and `implementer_id`,
`report` reference, `expert_status:"PENDING_EXPERT_REVIEW"`, and `findings` list.
Critical/high/P0/P1 findings require `FIXED_VALIDATED` or
`DISMISSED_WITH_EVIDENCE` and nonempty hashed evidence. Lowercase severity cannot
bypass this gate. Human credentials and independence remain producer assertions;
the validator does not invent expert labels or authenticate personnel.

Protected reference names are `final3_wheel`, `final3_sdist`, `acceptance_report`,
`closeout_audit`, `release_manifest`, `delivery_validation`, and
`sealed_holdout_manifest`. All seven hashes are pinned to the preserved final3
authority. The sealed manifest is only hashed; no sealed evaluation is executed.

Git closure requires a clean current branch, exact local checkpoint equality,
functional-source ancestry, and a retained successful `git ls-remote` command for
the configured remote and exact `refs/heads/<remote_branch>`. Recorded remote SHA
must equal current HEAD, and the observation must postdate that commit. Remote
equality is asserted at the recorded observation time; the validator makes no
network request and cannot guarantee the server has not changed afterward.

## Validation scope

`tests/test_final_campaign_acceptance.py` uses explicitly synthetic unit fixtures
to attack validator contracts, not as scientific validation evidence. Mutations
cover archive CRC/RECORD/path safety, missing receipts, hidden JUnit failures and
skips, stale source, incomplete matrices, pending closure, claim locators/spans,
installed import locations, network/origin/model replay, retrieval denominators,
unsupported expert status and open high review findings. Native final acquisition,
omics and benchmark receipts are also read and checked without rerunning their
scientific computations. Root integration supplies final executed install, CLI,
offline, full-suite and independent-review evidence after source assembly.

The delivered local gate passed 69 tests with zero failures, errors or skips:

```powershell
C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe -B -m pytest tests/test_final_campaign_acceptance.py -q -p no:cacheprovider --junitxml=C:\Codex\LitDataMatcher-v2\data\final_campaign_20260913\final_validator\delivery_checks\fixture_tests.xml
```

Exact command/source-state, JUnit, CLI and static-check receipts are under
`C:\Codex\LitDataMatcher-v2\data\final_campaign_20260913\final_validator\delivery_checks`.
The checkout wrapper generated the ledger example with exit 0; the installed-style
module entry point rejected that deliberately incomplete ledger with exit 1.
Ruff passed. Python 3.10 grammar was checked; execution used the existing Python
3.12 environment, so this is not a claim of a separate Python 3.10 runtime test.
Earlier native receipt checks under this evidence root passed acquisition
(1600 literature / 487 dataset IDs / 37 snapshots), omics (65 IDs / 3 snapshots),
and real benchmark contracts (17 queries x 552 candidates, three scale points,
recovery exits 71/0). Their original source bindings remain recorded; root will
execute the final unchanged-protocol benchmark derivative on assembled source.

The subsequent Windows build exposed a metadata-header parsing defect: valid
CRLF `Name` and `Version` fields retained a trailing carriage return under the
original regular expression. Distribution inspection now uses the standard
library email parser on headers only. Each metadata block must have exactly one
nonempty Name and Version field; duplicate fields are rejected case-insensitively,
including identical duplicates. Description-body examples cannot supply or
override either field. Cross-file package identity/version checks and wheel
RECORD, archive CRC, member safety, and source comparisons remain in force.

The retained repair receipts are under
`C:\Codex\LitDataMatcher-v2\data\final_campaign_20260913\final_validator\metadata_header_repair_20260913`.
They record 16 failing pre-repair regressions and reopen the original failed-build
wheel and sdist without extraction or alteration. The post-repair fixture suite
and real artifact results are recorded separately in `green_after.json` and its
JUnit/stdout artifacts. This is parser validation, not a replacement final build.
