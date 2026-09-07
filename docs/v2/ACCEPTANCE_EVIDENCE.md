# Acceptance evidence ledger

Run the validator only after preserving the run evidence it will inspect:

```powershell
litdatamatcher-v2 acceptance --evidence C:/path/to/ACCEPTANCE_EVIDENCE.json --out C:/path/to/ACCEPTANCE_REPORT.json
```

The ledger is a JSON object with schema version `1.0`. It lists one or more
run manifests and the executed checks they substantiate. Paths are relative to
the ledger and run-manifest directories; absolute paths and traversal are
rejected.

```json
{
  "schema_version": "1.0",
  "build_id": "litdatamatcher-v2-20260907",
  "runs": [
    {
      "run_manifest": "runs/real-run/RUN_MANIFEST.json",
      "checks": [
        {
          "id": "fresh-runtime-001",
          "target": "G05",
          "kind": "fresh_application_process",
          "command_index": 0,
          "artifacts": ["inferences.jsonl"],
          "observed_at": "2026-09-07T12:00:00+00:00"
        }
      ]
    }
  ],
  "open_issues": [],
  "optional_backlog": [],
  "stop_reason": null
}
```

Each check must use one of the gate or operation evidence kinds defined by the
validator. A PASS requires every kind for the target. Each referenced run must
be schema-valid, have `execution_status: PASS`, record no failures, and include
a successful command whose nonempty log and every cited artifact are listed as
`validation: PASS` artifacts with current matching SHA-256 and byte count. The
observation timestamp must fall within the recorded run interval.

The report is derived from those validated records. A README, a nonempty file,
a hand-written report, an unverified count, a stale observation, or a failed
command is never enough to promote a gate. Missing required records are
reported as `NOT_RUN`; inconsistent records are reported as `FAIL`.
